import torch
import time
import numpy as np
import cv2
import os
from torchvision import transforms
from utils import add_mask_to_source_multi_classes
from pathlib import Path
from dataset import SegDataset
from choices import get_criterion
from matplotlib import pyplot as plt
from metric import SegmentationMetric
import json
from collections import Counter
from tqdm import tqdm
from scipy import interpolate
from PIL import Image
from collections import namedtuple
import torch.utils.data as data
import torch.nn.functional as F

def miouandpa(config, output, target):
    target = target.squeeze(1)
    pa, miou = None, None

    metric = SegmentationMetric(config.out_channels)
    metric.update(output, target)
    pa, miou = metric.get_multigpu()

    return miou, pa

def predict_one_image(net, out_channels, batch_data, batch_label):
    if batch_label is None:
        batch_label = torch.randn(batch_data.shape[0], out_channels, batch_data.shape[2], batch_data.shape[3])
    with torch.no_grad():
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        output = net(batch_data)
        if out_channels == 1:
            batch_label = batch_label.float()
            output = torch.sigmoid(output).squeeze().cpu()
            prediction_np = np.where(np.array(output) > 0.5, 1, 0)
        else:
            batch_label = batch_label.squeeze(1)  # 交叉熵损失需要去掉通道维C
            prediction_max = torch.max(output.data, 1)[1].squeeze(0).cpu()
            prediction_mean = torch.mean(output.data, 1).squeeze(0).cpu()

        return prediction_max, prediction_mean

def predict_one_image_for_one_layer(layer_name,net, out_channels, batch_data, batch_label):
    features_in_hook = []
    features_out_hook = []
    def hook(module, fea_in, fea_out):
        features_in_hook.append(fea_in)
        features_out_hook.append(fea_out)
        return None
    for (name, module) in net.named_modules():
        print(name)
        if name == layer_name:
            module.register_forward_hook(hook=hook)

    if batch_label is None:
        batch_label = torch.randn(batch_data.shape[0], out_channels, batch_data.shape[2], batch_data.shape[3])
    with torch.no_grad():
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        output = net(batch_data)
        output_layer = features_out_hook[-1]
        if out_channels == 1:
            batch_label = batch_label.float()
            output = torch.sigmoid(output).squeeze().cpu()
            prediction_np = np.where(np.array(output) > 0.5, 1, 0)
        else:
            batch_label = batch_label.squeeze(1)  # 交叉熵损失需要去掉通道维C
            prediction_max = torch.max(output_layer.data, 1)[1].squeeze(0).cpu()
            prediction_mean = torch.mean(output_layer.data, 1).squeeze(0).cpu()
        return prediction_max, prediction_mean


def predict_a_batch(net, out_channels, batch_data, batch_label, class_weights, do_criterion, do_metric):
    if batch_label is None:
        batch_label = torch.randn(batch_data.shape[0], out_channels, batch_data.shape[2], batch_data.shape[3])
    with torch.no_grad():
        batch_data = batch_data.cuda()
        batch_label = batch_label.cuda()
        output = net(batch_data)
        if out_channels == 1:
            batch_label = batch_label.float()
            output = torch.sigmoid(output).squeeze().cpu()
            prediction_np = np.where(np.array(output) > 0.5, 1, 0)
        else:
            batch_label = batch_label.squeeze(1)

            prediction_np = np.array(torch.max(output.data, 1)[1].squeeze(0).cpu())

        loss, pa, miou = None, None, None

        criterion = get_criterion(out_channels, class_weights)
        if do_criterion:
            loss = criterion(output.cuda(), batch_label).item()

        if do_metric:
            metric = SegmentationMetric(out_channels)
            metric.update(output, batch_label)
            # metric.update(output.cuda(), batch_label.cuda())
            pa, miou = metric.get()

        return prediction_np, loss, (pa, miou)


def eval_dataset_full(net, out_channels, loader, class_weights=None, save_dir=None):
    mious, pas, losses, batch_data_shape = [], [], [], ()
    for i, (batch_data, batch_label) in enumerate(loader):
        if i == 0:
            batch_data_shape = batch_data.shape
        _, loss, (pa, miou) = predict_a_batch(net, out_channels, batch_data, batch_label, class_weights=class_weights,
                                              do_criterion=True, do_metric=True)
        losses.append(loss)
        mious.append(miou)
        pas.append(pa)
        print('Predicted batch [{}/{}], Loss:{}, IoU:{}, PA:{}'.format(i, len(loader), round(loss, 3), round(miou, 3),
                                                                       round(pa, 3)))

    mean_iou = round(float(np.mean(mious)), 3)
    pixel_acc = round(float(np.mean(pas)), 3)
    avg_loss = round(float(np.mean(losses)), 3)
    print('Average loss:{}, Mean IoU:{}, Pixel accuracy:{}'.format(avg_loss, mean_iou, pixel_acc))
    if save_dir is None:
        return avg_loss, (mean_iou, pixel_acc)
    else:
        from ptflops import get_model_complexity_info
        image = (batch_data_shape[1], batch_data_shape[2], batch_data_shape[3])
        GFLOPs, Parameters = get_model_complexity_info(net.cuda(), image, as_strings=True, print_per_layer_stat=False,
                                                       verbose=False)
        save_dict = {}
        save_dict.setdefault('GFLOPs', GFLOPs)
        save_dict.setdefault('Parameters', Parameters)
        save_dict.setdefault('Average loss', avg_loss)
        save_dict.setdefault('Mean IoU', mean_iou)
        save_dict.setdefault('Pixel accuracy', pixel_acc)
        with open(save_dir + '/metrics.json', 'w') as f:
            import json
            json.dump(save_dict, f, indent=2)
        with open(save_dir+'/test_loss.txt', 'a') as t:
            for l in losses:
                t.writelines(str(round(l,3)) + '\n')
        with open(save_dir+'/test_miou.txt', 'a') as t:
            for l in mious:
                t.writelines(str(round(l,3)) + '\n')
        with open(save_dir+'/test_pa.txt', 'a') as t:
            for l in pas:
                t.writelines(str(round(l,3)) + '\n')




def predict_images(net, args, dst_size=(512, 512), save_dir=None):
    if not args.test_images:
        print('Test image path is not specific!')
        return

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    pred_dicts = []

    times = []
    paths = [i for i in Path(args.test_images).glob('*.jpg')]
    for path in paths:
        frame = cv2.imread(str(path))
        start = time.time()

        img_transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((args.height, args.width)), transforms.ToTensor()])
        img_tensor = img_transform(frame).unsqueeze(0)
        prediction_np, _, _ = predict_a_batch(net=net, out_channels=args.out_channels, batch_data=img_tensor,batch_label=None, class_weights=None,
                                              do_criterion=False, do_metric=False)
        prediction_np = prediction_np.astype('uint8')
        if args.erode > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.erode, args.erode))
            prediction_np = cv2.erode(prediction_np, kernel)
        dst_frame = cv2.resize(frame, dst_size)
        dst_prediction = cv2.resize(prediction_np, dst_size)

        pred_dict = Counter(dst_prediction.flatten())
        pred_dict2 = {'path': path.name, 0: pred_dict[0], 1: pred_dict[1], 2: pred_dict[2], 3: pred_dict[3]}
        pred_dicts.append(pred_dict2)
        print(pred_dict2)
        dst_show = add_mask_to_source_multi_classes(dst_frame, dst_prediction, args.out_channels)

        torch.cuda.synchronize()
        end = time.time()
        cost_time = end - start
        times.append(cost_time)
        print('Processed image:{}\t\tTime:{}'.format(path.name, cost_time))
        if save_dir is not None:
            cv2.imwrite(save_dir + '/test_image-' + args.pt_dir + '-' + path.name + '.jpg', dst_show)
        else:
            plt.imshow(dst_show)
            plt.pause(0.5)




def feature_vis_mean(outshape, feats, savedir, iter):
    output_shape = outshape
    channel_mean = feats  # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
    channel_mean = channel_mean.cpu().detach().numpy()
    channel_mean = (
                ((channel_mean - np.min(channel_mean)) / (np.max(channel_mean) - np.min(channel_mean))) * 255).astype(
        np.uint8)
    savedir = savedir
    if not os.path.exists(savedir): os.makedirs(savedir)
    channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
    cv2.imwrite(savedir + '/{}.png'.format(iter), channel_mean)


def feature_vis_max(outshape, feats, savedir, iter):
    output_shape = outshape
    channel_mean = feats
    channel_mean = channel_mean.cpu().detach().numpy()
    channel_mean = (
            ((channel_mean - np.min(channel_mean)) / (np.max(channel_mean) - np.min(channel_mean))) * 255).astype(
        np.uint8)
    savedir = savedir
    if not os.path.exists(savedir): os.makedirs(savedir)
    channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
    cv2.imwrite(savedir + '/{}.png'.format(iter), channel_mean)

def predict_Attention_images(net, args, dst_size=(512, 512), save_dir=None, model_name=None):
    if not args.test_images:
        print('Test image path is not specific!')
        return
    save_dir = os.path.join(save_dir, model_name)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    creat_dir = os.path.join(save_dir, "outc")
    if creat_dir is not None:
        if not os.path.exists(creat_dir):
            os.makedirs(creat_dir)
    creat_dir = os.path.join(save_dir, "inc")
    if creat_dir is not None:
        if not os.path.exists(creat_dir):
            os.makedirs(creat_dir)
    creat_dir = os.path.join(save_dir, "down1")
    if creat_dir is not None:
        if not os.path.exists(creat_dir):
            os.makedirs(creat_dir)
    creat_dir = os.path.join(save_dir, "down2")
    if creat_dir is not None:
        if not os.path.exists(creat_dir):
            os.makedirs(creat_dir)
    creat_dir = os.path.join(save_dir, "down3")
    if creat_dir is not None:
        if not os.path.exists(creat_dir):
            os.makedirs(creat_dir)
    creat_dir = os.path.join(save_dir, "down4")
    if creat_dir is not None:
        if not os.path.exists(creat_dir):
            os.makedirs(creat_dir)

    pred_dicts = []
    times = []
    paths = [i for i in Path(args.test_images).glob('*.jpg')]
    filename = 'outc'
    # search [inc down1 down2 down3 down4 outc]
    if filename == 'down4':
        outshape = (32,32)
        save_dir = os.path.join(save_dir, filename)
    elif filename == 'down3':
        outshape = (64,64)
        save_dir = os.path.join(save_dir, filename)
    elif filename == 'down2':
        outshape = (128,128)
        save_dir = os.path.join(save_dir, filename)
    elif filename == 'down1':
        outshape = (256,256)
        save_dir = os.path.join(save_dir, filename)
    elif filename == 'inc':
        outshape = (512,512)
        save_dir = os.path.join(save_dir, filename)
    else:
        outshape = (512, 512)
        save_dir = os.path.join(save_dir, filename)
    for path in paths:
        name = path.name.split('.')
        frame = cv2.imread(str(path))
        start = time.time()
        img_transform = transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize((args.height, args.width)), transforms.ToTensor()])
        img_tensor = img_transform(frame).unsqueeze(0)

        if filename == 'outc':
            prediction_max, prediction_mean = predict_one_image(net=net, out_channels=args.out_channels, batch_data=img_tensor, batch_label=None)
            feature_vis_max(outshape=outshape, feats=prediction_max, savedir=save_dir,
                             iter=model_name + 'max_out' + name[0])
            feature_vis_mean(outshape=outshape,feats=prediction_mean, savedir=save_dir, iter=model_name+'mean_out' + name[0])

        else:
            prediction_layer_max, prediction_layer_mean = predict_one_image_for_one_layer(layer_name=filename, net=net, out_channels=args.out_channels,
                                                                batch_data=img_tensor, batch_label=None)
            feature_vis_max(outshape=outshape, feats=prediction_layer_max, savedir=save_dir,
                             iter=model_name + "max" + filename + name[0])
            feature_vis_mean(outshape=outshape,feats=prediction_layer_mean, savedir=save_dir, iter=model_name + "min" +filename + name[0])