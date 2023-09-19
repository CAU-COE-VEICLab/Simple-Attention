
from Nets.CSNet import CSNet
from Nets.munet import MUNet
from Nets.cpunet import CPUNet


import torch
import torch.nn as nn

from torchsummary import summary

from lr_scheduler import *


def get_lr_scheduler(optimizer, max_iters, sch_name):
    if sch_name == 'warmup_poly':
        return WarmupPolyLR(optimizer, max_iters=max_iters, power=0.9, warmup_factor=float(1.0/3), warmup_iters=0, warmup_method='linear')
    else:
        return None


def get_optimizer(net, optim_name):
    if optim_name == 'adam':
        optimizer = torch.optim.Adam(net.parameters())
    elif optim_name == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(net.parameters())
    return optimizer


def get_criterion(out_channels, class_weights=None):
    if out_channels == 1:
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    return criterion


def choose_net(name, out_channels):
    if name == 'CNNSimpleAttention':
        return CSNet(n_classes=out_channels, num_heads=2, drop_path_rate=0.1,choice='CNNSimpleAttention', patch_size=16)
    elif name == 'SimpleAttention':
        return CSNet(n_classes=out_channels, num_heads=2, drop_path_rate=0.1,choice='SimpleAttention', patch_size=16)
    elif name == 'munet':
        return MUNet(n_classes=out_channels, drop_path_rate=0.1)
    elif name == 'cpunet':
        return CPUNet(n_classes=out_channels, drop_path_rate=0.1)


