from einops.layers.torch import Rearrange
from einops import rearrange
from os.path import join as pjoin
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

device = torch.device('cuda:0')
torch.cuda.set_device(device)

PATCH_SIZE = 1
IMG_SIZE = 14
IMG_CHANNEL = 1024
TOKEN_NUM = (IMG_SIZE // PATCH_SIZE) ** 2 + 1
ATTENTION_HEAD_NUM = 4
EMBEDDING_DEPTH = 256
BATCH_SIZE = 2


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(1, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


class Patch_Embadding(nn.Module):
    def __init__(self):
        super(Patch_Embadding, self).__init__()
        self.patch_size = PATCH_SIZE
        self.projection_channel = EMBEDDING_DEPTH
        self.img_size = IMG_SIZE
        self.img_channel = IMG_CHANNEL
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=self.projection_channel, kernel_size=self.patch_size,
                      stride=self.patch_size),
            Rearrange('b e w h -> b (w h) e')
        )
        self.cls_token = nn.Parameter(torch.randn(BATCH_SIZE, 1, self.projection_channel))
        self.position_token = nn.Parameter(
            torch.randn((self.img_size // self.patch_size) ** 2 + 1, self.projection_channel))

    def forward(self, tensor):
        tensor = self.projection(tensor)
        # cls_token = repeat(self.cls_token,'() n e -> b n e',b=BATCH_SIZE)
        tensor = torch.cat([tensor, self.cls_token], dim=1)
        tensor += self.position_token
        return tensor


class Multi_Head_Attention(nn.Module):
    def __init__(self):
        super(Multi_Head_Attention, self).__init__()
        self.q = nn.Linear(in_features=EMBEDDING_DEPTH, out_features=EMBEDDING_DEPTH)
        self.k = nn.Linear(in_features=EMBEDDING_DEPTH, out_features=EMBEDDING_DEPTH)
        self.v = nn.Linear(in_features=EMBEDDING_DEPTH, out_features=EMBEDDING_DEPTH)
        self.att_drop = nn.Dropout(0.5)
        self.projection = nn.Linear(in_features=EMBEDDING_DEPTH, out_features=EMBEDDING_DEPTH)

    def forward(self, tensor):
        q = rearrange(self.q(tensor), 'b n (h d) -> b h n d', b=BATCH_SIZE, h=ATTENTION_HEAD_NUM)
        k = rearrange(self.k(tensor), 'b n (h d) -> b h n d', b=BATCH_SIZE, h=ATTENTION_HEAD_NUM)
        v = rearrange(self.v(tensor), 'b n (h d) -> b h n d', b=BATCH_SIZE, h=ATTENTION_HEAD_NUM)

        energy = torch.einsum('b h q d , b n k d -> b h q k ', q, k)
        att = F.softmax(energy, dim=-1) / EMBEDDING_DEPTH ** (1 / 2)
        att = self.att_drop(att)
        out = torch.matmul(att, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        out += tensor
        return out


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=EMBEDDING_DEPTH, out_features=EMBEDDING_DEPTH * 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=EMBEDDING_DEPTH * 2, out_features=EMBEDDING_DEPTH)
        )

    def forward(self, tensor):
        x = self.mlp(tensor)
        x += tensor
        return x


class ENCODER_BLOCK(nn.Module):
    def __init__(self):
        super(ENCODER_BLOCK, self).__init__()
        self.mha = Multi_Head_Attention()
        self.mlp = MLP()

    def forward(self, tensor):
        tensor = self.mha(tensor)
        tensor = self.mlp(tensor)
        return tensor


class MLP_HEAD(nn.Module):
    def __init__(self):
        super(MLP_HEAD, self).__init__()
        self.norm = nn.LayerNorm(TOKEN_NUM * EMBEDDING_DEPTH)
        self.linear = nn.Linear(TOKEN_NUM * EMBEDDING_DEPTH, 38612)

    def forward(self, tensor):
        tensor = rearrange(tensor, 'b n d -> b (n d)')
        tensor = self.norm(tensor)
        tensor = self.linear(tensor)
        return tensor


class VIT(nn.Module):
    def __init__(self, encoder_block_num=8):
        super(VIT, self).__init__()
        self.block_num = encoder_block_num
        self.patch_embedding = Patch_Embadding()
        self.encoder_block = ENCODER_BLOCK()
        self.mlp_head = MLP_HEAD()

    def forward(self, tensor):
        tensor = self.patch_embedding(tensor)
        for i in range(self.block_num):
            tensor = self.encoder_block(tensor)
        tensor = self.mlp_head(tensor)
        return tensor


class TransUnet(nn.Module):
    def __init__(self):
        super(TransUnet, self).__init__()
        self.vit = VIT()
        self.resnet = ResNetV2(block_units=(3, 4, 6), width_factor=1)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=197, out_channels=512, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU()
        )
        self.block2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, bias=True)

        self.block2_2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.block2_3 = nn.BatchNorm2d(num_features=256)
        self.block2_4 = nn.ReLU(inplace=True)

        self.block3_1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, bias=True)

        self.block3_2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.block3_3 = nn.BatchNorm2d(num_features=128)
        self.block3_4 = nn.ReLU(inplace=True)

        self.block4_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, bias=True)

        self.block4_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block4_3 = nn.BatchNorm2d(num_features=64)
        self.block4_4 = nn.ReLU(inplace=True)

        self.block5_1 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=2, stride=2, bias=True)
        self.block5_2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding='same')
        self.block5_3 = nn.BatchNorm2d(num_features=1)
        self.block5_4 = nn.ReLU(inplace=True)

    def forward(self, tensor):
        proj, skip = self.resnet(tensor)

        def vit_checkpoint(proj):
            return self.vit(proj)

        before_upconv = checkpoint(vit_checkpoint, proj)
        before_upconv = torch.reshape(before_upconv, (BATCH_SIZE, TOKEN_NUM, 14, 14))
        tensor = self.block1(before_upconv)

        tensor = self.block2_1(tensor)
        tensor = torch.concat([tensor, skip[0]], dim=1)
        tensor = self.block2_2(tensor)
        tensor = self.block2_3(tensor)
        tensor = self.block2_4(tensor)

        tensor = self.block3_1(tensor)
        tensor = torch.concat([tensor, skip[1]], dim=1)
        tensor = self.block3_2(tensor)
        tensor = self.block3_3(tensor)
        tensor = self.block3_4(tensor)

        tensor = self.block4_1(tensor)
        tensor = torch.concat([tensor, skip[2]], dim=1)
        tensor = self.block4_2(tensor)
        tensor = self.block4_3(tensor)
        tensor = self.block4_4(tensor)

        tensor = self.block5_1(tensor)
        tensor = self.block5_2(tensor)
        tensor = self.block5_3(tensor)
        tensor = self.block5_4(tensor)

        return tensor


if __name__ == "__main__":
    dummy = torch.randn(size=(BATCH_SIZE, 1, 224, 224)).to('cuda')
    model = TransUnet().to('cuda')
    pred = model(dummy)
    print()
