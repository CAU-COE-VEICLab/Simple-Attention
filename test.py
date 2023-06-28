from pathlib import Path
from argparse import ArgumentParser
import torch
import json
from choices import choose_net
from predictor import eval_dataset_full, predict_images
from dataset import SegDataset, get_class_weights
from torch.utils.data import DataLoader
import os


def get_test_args():
    parser = ArgumentParser()
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--out-channels", type=int)
    parser.add_argument("--erode", type=int, default=0)
    parser.add_argument("--pt-dir", type=str)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--test-videos", type=str)
    parser.add_argument("--test-set", type=str)
    parser.add_argument("--test-images", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--weighting", type=str, default='none')
    parser.add_argument("--pt-root", type=str, default='./Results/')
    parser.add_argument("--vis", type=bool, default=True)
    return parser.parse_args()


def find_latest_pt(dir):
    best_path = ''
    for path in Path(dir).glob('*.pt'):
        if path.name.split('_')[-1].split('.')[0] == 'best':
            best_path = str(path)
    if best_path != '':
        return best_path
    else:
        print('No pts this dir:', dir)
        return None


def merge_args_from_train_json(args, json_path, verbose=False):
    if not os.path.exists(json_path):
        return args
    with open(json_path, 'r') as f:
        train_d = json.load(f)
        if verbose:
            print(train_d)
    args.weighting = train_d['weighting']
    args.dilate = train_d['erode']
    args.net_name = train_d['net_name']
    args.out_channels = train_d['out_channels']
    args.save_suffix = train_d['save_suffix']
    args.height = train_d['height']
    args.width = train_d['width']
    with open(json_path.replace('train_args', 'test_args'), 'w') as f:
        d = vars(args)
        json.dump(d, f, indent=2)
    if verbose:
        for k, v in d.items():
            print(k, v)
    return args


def do_test(mode, args):
    print('\nTesting: {}. ################################# Mode: {}'.format(args.pt_dir, mode))
    pt_dir = args.pt_root + '/' + args.pt_dir
    args = merge_args_from_train_json(args, json_path=pt_dir + '/train_args.json')
    pt_path = find_latest_pt(pt_dir)
    if pt_path is None:
        return
    print('Loading:', pt_path)
    net = choose_net(args.net_name, args.out_channels).cuda()
    net.load_state_dict(torch.load(pt_path))
    net.eval()
    test_loader, class_weights = None, None
    if mode == 0 or mode == 1:
        test_set = SegDataset(args.test_set, args.out_channels, appoint_size=(args.height, args.width), erode=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        class_weights = get_class_weights(test_loader, args.out_channels, args.weighting)

    if mode == 0:
        eval_dataset_full(net, args.out_channels, test_loader, class_weights=class_weights, save_dir=pt_dir)

    elif mode == 1:
        predict_images(net, args, dst_size=(512, 512), save_dir=args.save_dir)




def do_search(args, task=0):
    args.pt_root = './Result'
    pt_dirs = []

    if task == 0:
        args.test_videos = 'none'
        args.test_set = 'Data/aug/test'
        args.save_dir = './predict'
        args.test_images = 'Data/aug/test/images'

        args.out_channels = 4
        pt_dirs = [i.name for i in Path(args.pt_root).iterdir() if i.is_dir() and 'aug' in i.name]

    modes = [0, 1]
    for mode in modes:
        for pt_dir in pt_dirs:
            args.pt_dir = pt_dir
            with torch.cuda.device(0):
                do_test(mode, args)


if __name__ == "__main__":
    args = get_test_args()
    search_experiment = True
    if search_experiment:
        do_search(args)
    else:
        mode = 1
        do_test(mode, args)