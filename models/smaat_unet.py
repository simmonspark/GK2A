import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import init
import torch.nn.functional as F

'''
기존 unet과 차이점 
0. 모든 컨볼루션은 depth wise conv로 
1. encoder -> cbam 추가 doubble conv 이후에 cbam으로 어텐션 하고, residual을 위해 저장 
2. conv에서 depth wise conv로 바꿈 

loss 는 mse 씀 이상 정리 끝 


'''
IMG_C = 1
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 0, stride=1 ,bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size,stride=stride, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels,
                mid_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(
                mid_channels,
                out_channels,
                kernel_size=3,
                kernels_per_layer=kernels_per_layer,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
class UpDS(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvDS(
                in_channels,
                out_channels,
                in_channels // 2,
                kernels_per_layer=kernels_per_layer,
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class SMAT_unet(nn.Module):
    def __init__(self):
        super(SMAT_unet,self).__init__()
        self.dic_bam = [64,128,256,512,512]
        self.dic_conv = [1,32,64,96,128,176,256,384,512,512,512]
        self.for_skip = []

        self.cbam_1 = CBAMBlock(channel=self.dic_bam[0])
        self.cbam_2 = CBAMBlock(channel=self.dic_bam[1])
        self.cbam_3 = CBAMBlock(channel=self.dic_bam[2])
        self.cbam_4 = CBAMBlock(channel=self.dic_bam[3])
        self.cbam_5 = CBAMBlock(channel=self.dic_bam[4])


        self.max_pool = nn.MaxPool2d(kernel_size=(2))


        self.E_1 = nn.Sequential(OrderedDict(
            [
                (f'conv1',depthwise_separable_conv(nin=self.dic_conv[0],nout=self.dic_conv[1],stride=1,padding='same')),
                (f'conv2', depthwise_separable_conv(nin=self.dic_conv[1], nout=self.dic_conv[2],padding='same')),
                (f'BN', nn.BatchNorm2d(num_features=self.dic_conv[2])),
                (f'RELU', nn.ReLU())
            ]
        ))
        self.E_2 = nn.Sequential(OrderedDict(
            [
                (f'conv1', depthwise_separable_conv(nin=self.dic_conv[2], nout=self.dic_conv[3],stride=1,padding='same')),
                (f'conv2', depthwise_separable_conv(nin=self.dic_conv[3], nout=self.dic_conv[4],padding='same')),
                (f'BN', nn.BatchNorm2d(num_features=self.dic_conv[4])),
                (f'RELU', nn.ReLU())
            ]
        ))
        self.E_3 = nn.Sequential(OrderedDict(
            [
                (f'conv1', depthwise_separable_conv(nin=self.dic_conv[4], nout=self.dic_conv[5],stride=1,padding='same')),
                (f'conv2', depthwise_separable_conv(nin=self.dic_conv[5], nout=self.dic_conv[6],padding='same')),
                (f'BN', nn.BatchNorm2d(num_features=self.dic_conv[6])),
                (f'RELU', nn.ReLU())
            ]
        ))
        self.E_4 = nn.Sequential(OrderedDict(
            [
                (f'conv1', depthwise_separable_conv(nin=self.dic_conv[6], nout=self.dic_conv[7],stride=1,padding='same')),
                (f'conv2', depthwise_separable_conv(nin=self.dic_conv[7], nout=self.dic_conv[8],padding='same')),
                (f'BN', nn.BatchNorm2d(num_features=self.dic_conv[8])),
                (f'RELU', nn.ReLU())
            ]
        ))
        self.E_5 = nn.Sequential(OrderedDict(
            [
                (f'conv1', depthwise_separable_conv(nin=self.dic_conv[8], nout=self.dic_conv[9],stride=1,padding='same')),
                (f'conv2', depthwise_separable_conv(nin=self.dic_conv[9], nout=self.dic_conv[10],padding='same')),
                (f'BN', nn.BatchNorm2d(num_features=self.dic_conv[10])),
                (f'RELU', nn.ReLU())
            ]
        ))

        factor = 2
        kernels_per_layer = 2
        self.bilinear = True
        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.out = depthwise_separable_conv(nin=64, nout=1, kernel_size=1)



    def forward(self,tensor):

        tensor = self.E_1(tensor)
        skip1 = self.cbam_1(tensor)
        tensor = self.max_pool(tensor)


        tensor = self.E_2(tensor)
        skip2 = self.cbam_2(tensor)
        tensor = self.max_pool(tensor)

        tensor = self.E_3(tensor)
        skip3 = self.cbam_3(tensor)
        tensor = self.max_pool(tensor)

        tensor = self.E_4(tensor)
        skip4 = self.cbam_4(tensor)
        tensor = self.max_pool(tensor)
        tensor = self.E_5(tensor)
        tensor = self.cbam_5(tensor)

        tensor = self.up1(tensor, skip4 )
        tensor = self.up2(tensor, skip3)
        tensor = self.up3(tensor, skip2)
        tensor = self.up4(tensor, skip1)
        tensor = self.out(tensor)


        return tensor




if __name__ == "__main__":
    dummy = torch.randn(size=(4,1,288,288))
    model = SMAT_unet()
    pred = model(dummy)
    print()




