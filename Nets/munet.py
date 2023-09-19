import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"Droprate={self.drop_prob}"

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, drop_path_rate=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_path_rate)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop_path_rate=0.1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, drop_path_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, drop_path_rate=0.1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, drop_path_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Respath(nn.Module):
    def __init__(self, in_channels):
        super(Respath, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv(x)
        return skip+x

class RespathBlock(nn.Module):
    def __init__(self, in_channels):
        super(RespathBlock, self).__init__()
        self.respath = nn.Sequential(
            Respath(in_channels),
            Respath(in_channels),
            Respath(in_channels),
            Respath(in_channels),
        )

    def forward(self, x):
        return self.respath(x)

class MUNet(nn.Module):
    def __init__(self, n_classes, bilinear=True, drop_path_rate=0.1):
        super(MUNet, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(3, 64, drop_path_rate)
        self.down1 = Down(64, 128, drop_path_rate)
        self.down2 = Down(128, 256, drop_path_rate)
        self.down3 = Down(256, 512, drop_path_rate)
        self.down4 = Down(512, 512, drop_path_rate)
        self.up1 = Up(1024, 256, bilinear, drop_path_rate)
        self.up2 = Up(512, 128, bilinear, drop_path_rate)
        self.up3 = Up(256, 64, bilinear, drop_path_rate)
        self.up4 = Up(128, 64, bilinear, drop_path_rate)
        self.outc = OutConv(64, n_classes)
        self.respath1 = RespathBlock(64)
        self.respath2 = RespathBlock(128)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, self.respath2(x2))
        x = self.up4(x, self.respath1(x1))
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from thop import profile
    from pytorch_model_summary import summary
    import time

    print(torch.__version__)
    net = MUNet(2).cuda()
    print(net)
    image = torch.rand(1, 3,  800, 600).cuda()

    f, p = profile(net, inputs=(image,))
    # f, p = summary(net, (image, time_step))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))

    s = time.time()
    with torch.no_grad():
        out = net(image, )

    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))

    print(out.shape)
    batch_size = 2
    num_batches = 80
    input_data = torch.randn(batch_size, 3, 800, 600).cuda()
    def process_image(image):
        return net(image)
    net.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in range(num_batches):
            output = process_image(input_data)
    # 计时结束
    end_time = time.time()
    # 计算吞吐量（每秒处理的图像数量）
    total_images_processed = batch_size * num_batches
    throughput = total_images_processed / (end_time - start_time)
    print(f"throughput: {throughput:.2f} Image/s")