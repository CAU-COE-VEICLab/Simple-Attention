import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, drop_path_rate=0. ):
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

class FeatureDownScale(nn.Module):
    def __init__(self, down_scale, in_c):
        super(FeatureDownScale, self).__init__()
        self.proj = nn.Conv2d(in_c, in_c, kernel_size=down_scale, stride=down_scale)
        self.norm = nn.LayerNorm(in_c)

    def forward(self, x):

        x = self.norm(self.proj(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x



class SimpleAttention(nn.Module):
    """[1,1,c]->[1,c]@[c,c]->[1,c]*[h*w,c]->[h,w,c]"""
    def __init__(self,
                 in_channels: int,
                 num_heads: int = 2
                 ):
        super(SimpleAttention, self).__init__()
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.scale: float = num_heads ** -0.5
        # Init layers
        self.norm = nn.LayerNorm(in_channels)
        self.v = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.a = nn.Linear(in_features=1, out_features=in_channels//num_heads, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.s = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding='same',groups=in_channels,bias=True)

    def forward(self, x):
        # x[B,C,H,W]
        _B, _C, _H, _W = x.shape
        # [B,C,H,W]->[B,C,1,1]
        # [B,C,1,1]->[B,C,1]
        # x_temp: [B,C,1]
        skip = x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x_temp = x.mean((2, 3), keepdim=True).reshape(_B, _C, -1)
        # score_map: [B,C,C/head]
        # reshape:[B,num_head,c//num_head,C/head]
        Ka = self.a(x_temp).reshape(_B, self.num_heads, _C//self.num_heads, -1)

        # x_temp:[B,C,1] -> permute X_K [B,1,C]
        # x_k: [B,1,C]
        Kv = x_temp.permute(0, 2, 1)
        # reshape:[B,1,num_head,embed_dim_per_head]
        # permute:[B,num_head,1,embed_dim_per_head]
        Kv = self.v(Kv).reshape(_B, 1, self.num_heads, -1).permute(0, 2, 1, 3)

        # score:[B,num_head,1,C/head]
        # transpose:[B,num_head,C/head,1]
        # reshape:[B,C,1,1]
        score = self.softmax(Kv @ Ka.transpose(-2, -1) * self.scale).transpose(1, 2).reshape(_B, -1, 1, 1)
        # x[B,C,H,W]
        # score[B,C,1,1]
        x = self.s(x) * score
        # x[B,C,H,W]
        return x + skip

class FFNModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 drop_path_rate=0.
                 ):
        super(FFNModule, self).__init__()
        self.DConv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same', groups=in_channels)
        self.DCex = nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1, padding='same')
        self.DCre = nn.Conv2d(4 * in_channels, in_channels, kernel_size=1, padding='same')
        self.norm = nn.LayerNorm(in_channels)
        self.gelu = nn.GELU()
        # defeat over-fitting
        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()


    def forward(self, x):
        # x[B,C,H,W]
        _B, _C, _H, _W = x.shape
        skip = x
        x = self.DCre(self.gelu(self.DCex(self.norm(self.DConv(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2))))
        return self.drop_path(x)+skip

class CNNBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super(CNNBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same')
        self.conv1x1end = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding='same')
        self.gelu = nn.GELU()  # 在DWextern后 Fconv_3后
        self.batchnorm = nn.BatchNorm2d(out_channels)  # 在Fconv_11后

    def forward(self, x):
        # x[B,C,H,W]
        _B, _C, _H, _W = x.shape
        x = self.conv1x1end(self.gelu(self.batchnorm(self.conv3x3(self.conv1x1(x)))))

        return x

class CSModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads: int = 32, drop_path_rate=0., choice='CNNSampleAttention', patch_size = 16):
        super(CSModule, self).__init__()
        if choice == 'CNNSampleAttention':
            self.cnnblock = CNNBlock(in_channels, out_channels)
            self.attention = SimpleAttention(out_channels, num_heads)
        elif choice == 'SampleAttention':
            self.cnn = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
            self.attention = SimpleAttention(out_channels, num_heads)
        self.ffn = FFNModule(out_channels, drop_path_rate)
        # Make skip path
        self.skip_path = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        # CNN block
        cnn_shortcut = x
        x = self.cnnblock(x)
        x = self.drop_path(x) + self.skip_path(cnn_shortcut)
        # Simple-former
        x = self.ffn(x)
        x = self.attention(x)
        return x

class DownCSModule(nn.Module):
    """
    in_channels: input feature map channels
    out_channels: the feature map channels after download operation.So  out_channels=2*in_channels
    """
    def __init__(self, in_channels, out_channels, num_heads: int = 32, drop_path_rate=0., choice='CNNSampleAttention',patch_size = 16):
        super(DownCSModule, self).__init__()
        self.down = FeatureDownScale(down_scale=2, in_c=in_channels)
        self.csmodule = CSModule(in_channels, out_channels, num_heads, drop_path_rate, choice, patch_size=patch_size)

    def forward(self, x):
        # x: [B,C,H,W]
        # down: [B,C,H/2,W/2]
        # dwBlock: [B,2C,H/2,W/2]
        x = self.down(x)
        x = self.csmodule(x)
        return x

class UpConvModule(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, drop_path_rate=0.1, bilinear=True):
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
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Out, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CSNet(nn.Module):
    def __init__(self, n_classes, num_heads: int = 2, drop_path_rate=0.1, choice='CNNSampleAttention', patch_size=16):
        super(CSNet, self).__init__()
        self.n_classes = n_classes

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]

        self.inc = CSModule(3, 64, num_heads, drop_path_rate,choice=choice,patch_size=patch_size)
        self.down1 = DownCSModule(64, 128, num_heads, dpr[0], choice=choice, patch_size=patch_size)  # in_channels, out_channels, num_heads: int = 32, drop_path_rate=0.
        self.down2 = DownCSModule(128, 256, num_heads, dpr[1], choice=choice, patch_size=patch_size)
        self.down3 = DownCSModule(256, 512, num_heads, dpr[2], choice=choice, patch_size=patch_size)
        self.down4 = DownCSModule(512, 512, num_heads, dpr[3], choice=choice, patch_size=patch_size)
        self.up1 = UpConvModule(1024, 256, dpr[0])  # in_channels, out_channels, num_heads: int = 32, drop_path_rate=0.
        self.up2 = UpConvModule(512, 128, dpr[1])
        self.up3 = UpConvModule(256, 64, dpr[2])
        self.up4 = UpConvModule(128, 64, dpr[3])
        self.outc = Out(64, n_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    from thop import profile
    import time

    print(torch.__version__)
    net = CSNet(n_classes=4, num_heads=2, drop_path_rate=0.3, choice='CNNSampleAttention', patch_size=16).cuda()
    print(net)
    image = torch.rand(1, 3, 512, 384).cuda()

    f, p = profile(net, inputs=(image,))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))

    s = time.time()
    with torch.no_grad():
        out = net(image, )

    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))

    print(out.shape)

