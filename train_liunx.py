import os
import json
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from dataset import SegDataset, get_class_weights
from choices import choose_net, get_criterion, get_optimizer, get_lr_scheduler
from predictor import eval_dataset_full, predict_images, miouandpa
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch.backends.cudnn as cudnn
import random
import numpy as np
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, save_checkpoint_guorun, \
    NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
import time
import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def get_train_args():
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--net-name", type=str)
    parser.add_argument("--save-suffix", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--out-channels", type=int)
    parser.add_argument("--erode", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--train-set", type=str)
    parser.add_argument("--val-set", type=str)
    parser.add_argument("--test-set", type=str)
    parser.add_argument("--test-images", type=str)
    parser.add_argument("--f16", type=bool, default=False)
    parser.add_argument("--train-aug", type=bool, default=True)
    parser.add_argument("--opt-name", type=str, default='adam')
    parser.add_argument("--sch-name", type=str, default='warmup_poly')
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--pt-dir", type=str)
    parser.add_argument("--pt-stride", type=int, default=20)
    parser.add_argument("--weighting", type=str, default='none')
    parser.add_argument("--eval", type=bool, default=True)
    parser.add_argument("--test", type=bool, default=False)

    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument("--save_dir", default='', type=str, help='local rank for DistributedDataParallel')
    parser.add_argument("--local_rank", default=6, type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--resume', default="", type=str, help='resume from checkpoint')
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def build_loader(args):
    dataset_train = SegDataset(args.train_set, num_classes=args.out_channels, appoint_size=(args.height, args.width),
                               erode=args.erode, aug=args.train_aug)
    print(
        f"local rank {args.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset. dataset length {len(dataset_train)}")
    if args.eval:
        dataset_val = SegDataset(args.val_set, num_classes=args.out_channels, appoint_size=(args.height, args.width),
                                 erode=0)
        print(
            f"local rank {args.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset. dataset length {len(dataset_val)}")

    dataset_test = SegDataset(args.test_set, num_classes=args.out_channels, appoint_size=(args.height, args.width),erode=0)
    print(f"local rank {args.LOCAL_RANK} / global rank {dist.get_rank()} successfully build test dataset. dataset length {len(dataset_test)}")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train,
                                                    batch_size=args.batch_size,
                                                    num_workers=8,
                                                    pin_memory=True,
                                                    drop_last=True,
                                                    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test


def train(args, logger):
    local_rank = torch.distributed.get_rank()
    device = torch.device("cuda", local_rank)

    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test = build_loader(args)
    logger.info(f"Creating model:{args.net_name}")

    model = choose_net(args.net_name, args.out_channels)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    model.to(device)
    model_without_ddp = model
    logger.info(f"remove to gpu")

    optimizer = get_optimizer(model, args.opt_name)
    logger.info(f"load optimizer")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    # loss_scaler = NativeScalerWithGradNormCount()
    logger.info(f"load  ddp")
    criterion = torch.nn.CrossEntropyLoss()
    logger.info(f"load  CEL")
    max_miou=0
    if args.resume != '':
        logger.info(f"load  resume")
        max_miou = load_checkpoint(args, model_without_ddp, optimizer, logger)
        miou, pa, loss, memory_used = validate(args, data_loader_val, model, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {miou:.3f}%")
        if args.test:
            miou, pa, loss, memory_used=test(args, data_loader_test, model, logger)
            return
    logger.info(f"out  resume")
    if args.throughput:
        logger.info(f"use  throughput")
        throughput(data_loader_val, model, args)
        return

    logger.info("Start training")
    start_time = time.time()
    miou_list = []
    pa_list = []
    loss_list = []
    for epoch in range(args.start_epoch, args.epoch):
        data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(args, model, criterion, data_loader_train, optimizer, epoch, logger)
        if dist.get_rank() == 0 and (epoch % args.pt_stride == 0 or epoch == (args.epoch - 1)):
            save_checkpoint(args, epoch, model_without_ddp, max_miou, optimizer, logger)

        miou, pa, loss, memory_used = validate(args, data_loader_val, model, logger)
        logger.info(
            f"MIOU of the network on the {len(dataset_val)} test images: LOSS{loss:.3f} PA{pa:.3f} MIOU{miou:.3f} Memory used{memory_used:.0f}MB")

        max_miou = max(max_miou, miou)
        logger.info(f'Max accuracy: {max_miou:.3f}%')
        if max_miou == miou and dist.get_rank() == 0:
            save_checkpoint_guorun(args, epoch, model_without_ddp, max_miou, optimizer, logger)
        miou_list.append(miou)
        pa_list.append(pa)
        loss_list.append(loss)

    if dist.get_rank() == 0:
        save_metrics(args, miou_list, pa_list, loss_list, memory_used)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    miou, pa, loss, memory_used = test(args, data_loader_test, model, logger)
    logger.info(
        f"MIOU of the network on the {len(dataset_val)} test images: LOSS{loss:.3f} PA{pa:.3f} MIOU{miou:.3f} Memory used{memory_used:.0f}MB")


def save_metrics(config, mean_iou_list, pixel_acc_list, avg_loss_list, memory_used):
    save_dict = {}
    save_dict.setdefault('Average loss', avg_loss_list)
    save_dict.setdefault('Mean IoU', mean_iou_list)
    save_dict.setdefault('Pixel accuracy', pixel_acc_list)
    save_dict.setdefault('Memory used', memory_used)
    with open(config.save_dir + '/metrics.json', 'w') as f:
        import json
        json.dump(save_dict, f, indent=2)
    with open(config.save_dir + '/test_loss.txt', 'a') as t:
        for l in avg_loss_list:
            t.writelines(str(round(l, 3)) + '\n')
    with open(config.save_dir + '/test_miou.txt', 'a') as t:
        for l in mean_iou_list:
            t.writelines(str(round(l, 3)) + '\n')
    with open(config.save_dir + '/test_pa.txt', 'a') as t:
        for l in pixel_acc_list:
            t.writelines(str(round(l, 3)) + '\n')
    with open(config.save_dir + '/memory_used.txt', 'a') as t:
        t.writelines(str(round(memory_used, 0)) + '\n')


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, logger):
    model.train()
    optimizer.zero_grad()
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    local_rank = torch.distributed.get_rank()
    device = torch.device("cuda", local_rank)
    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        if config.out_channels == 1:
            targets = targets.float()  # 逻辑损失需要label的类型和data相同，均为float，而不是long
        else:
            targets = targets.squeeze(1)  # 交叉熵label的类型采用默认的long，但需要去除C通道维
        samples = samples.cuda(non_blocking=True).to(device)
        targets = targets.cuda(non_blocking=True).to(device)

        outputs = model(samples)
        loss = criterion(outputs.to(device), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        logger.info(
            f'Train: [{epoch}/{config.epoch}][{idx}/{num_steps}]\t'
            f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
            f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, logger):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    miou_meter = AverageMeter()
    pa_meter = AverageMeter()
    local_rank = torch.distributed.get_rank()
    device = torch.device("cuda", local_rank)
    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True).to(device)
        target = target.cuda(non_blocking=True).to(device)
        output = model(images)

        loss = criterion(output.to(device), target)
        miou, pa = miouandpa(config, output, target)

        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        miou_meter.update(miou, target.size(0))
        pa_meter.update(pa, target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        logger.info(
            f'Test: [{idx}/{len(data_loader)}]\t'
            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            f'MIOU {miou_meter.val:.3f} ({miou_meter.avg:.3f})\t'
            f'PA {pa_meter.val:.3f} ({pa_meter.avg:.3f})\t'
            f'Mem {memory_used:.0f}MB')
    logger.info(f' * MIOU {miou_meter.avg:.3f} PA {pa_meter.avg:.3f}')
    return miou_meter.avg, pa_meter.avg, loss_meter.avg, memory_used


@torch.no_grad()
def test(config, data_loader, model, logger):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    miou_meter = AverageMeter()
    pa_meter = AverageMeter()
    local_rank = torch.distributed.get_rank()
    device = torch.device("cuda", local_rank)
    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True).to(device)
        target = target.cuda(non_blocking=True).to(device)
        output = model(images)

        loss = criterion(output.to(device), target)
        miou, pa = miouandpa(config, output, target)

        loss = reduce_tensor(loss)

        loss_meter.update(loss, target.size(0))
        miou_meter.update(miou, target.size(0))
        pa_meter.update(pa.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        logger.info(
            f'Test: [{idx}/{len(data_loader)}]\t'
            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
            f'MIOU {miou_meter.val:.3f} ({miou_meter.avg:.3f})\t'
            f'PA {pa_meter.val:.3f} ({pa_meter.avg:.3f})\t'
            f'Mem {memory_used:.0f}MB')
    logger.info(f' * MIOU {miou_meter.avg:.3f} PA {pa_meter.avg:.3f}')
    return miou_meter.avg, pa_meter.avg, loss_meter.avg, memory_used


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()
    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


def do_train(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    print(rank, world_size)

    torch.cuda.set_device(args.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enable = True
    cudnn.benchmark = True
    logger = create_logger(output_dir=args.save_dir, dist_rank=dist.get_rank(), name=f"{args.net_name}")
    if dist.get_rank() == 0:
        # Prepare save dir
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        with open(args.save_dir + '/train_args.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
    train(args, logger)


def get_choices(args, task):
    sizes = [(512, 512)]
    weightings = ['none']
    if task == 0:
        erodes = [0]
        args.train_set = 'Data/aug-disease/train'
        args.val_set = 'Data/aug-disease/val'
        args.test_set = 'Data/aug-disease/test'

    return args, sizes, erodes, weightings


def search_train(args):
    args.out_channels = 2
    args.epoch = 120
    args.batch_size = 6
    args.gpu = 0
    args.LOCAL_RANK = 0

    train_net_names = ['CSNet']
    # "CNNSimpleAttention" "SimpleAttention" "munet" "cpunet"
    save_suffix = 'aug'
    task = 0
    args, sizes, erodes, weightings = get_choices(args, task=task)

    for net_name in train_net_names:
        for size in sizes:
            for erode in erodes:
                for weighting in weightings:
                    args.weighting = weighting
                    args.erode = erode
                    args.net_name = net_name
                    args.height = int(size[0])
                    args.width = int(size[1])
                    args.save_suffix = save_suffix
                    args.save_dir = '/ResultLi' + args.save_suffix + '-' + args.net_name + '-erode' + str(
                        args.erode) + '-weighting_' + str(args.weighting)
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    print(f'Save dir is:{args.save_dir}')
                    do_train(args)


if __name__ == "__main__":
    args = get_train_args()
    search_experiment = True
    if search_experiment:
        search_train(args)
    else:
        do_train(args)
