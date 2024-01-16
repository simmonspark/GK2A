import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange,repeat
import torch.nn.functional as F
#predifine param
PATCH_SIZE  = 32
IMG_SIZE = 224
IMG_CHANNEL = 1
TOKEN_NUM = (IMG_SIZE//PATCH_SIZE)**2 +1
ATTENTION_HEAD_NUM = 8
EMBEDDING_DEPTH = 1024
BATCH_SIZE = 8

class Patch_Embadding(nn.Module):
    def __init__(self):
        super(Patch_Embadding,self).__init__()
        self.patch_size = PATCH_SIZE
        self.projection_channel = EMBEDDING_DEPTH
        self.img_size = IMG_SIZE
        self.img_channel = IMG_CHANNEL
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=self.projection_channel,kernel_size=self.patch_size,stride=self.patch_size),
            Rearrange('b e w h -> b (w h) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,self.projection_channel))
        self.position_token = nn.Parameter(torch.randn((self.img_size//self.patch_size)**2 +1,self.projection_channel))

    def forward(self,tensor):
        tensor = self.projection(tensor)
        cls_token = repeat(self.cls_token,'() n e -> b n e',b=BATCH_SIZE)
        tensor = torch.cat([tensor,cls_token],dim=1)
        tensor += self.position_token
        return tensor

class Multi_Head_Attention(nn.Module):
    def __init__(self):
        super(Multi_Head_Attention,self).__init__()
        self.q = nn.Linear(in_features=EMBEDDING_DEPTH,out_features=EMBEDDING_DEPTH)
        self.k = nn.Linear(in_features=EMBEDDING_DEPTH,out_features=EMBEDDING_DEPTH)
        self.v = nn.Linear(in_features=EMBEDDING_DEPTH,out_features=EMBEDDING_DEPTH)
        self.att_drop = nn.Dropout(0.5)
        self.projection = nn.Linear(in_features=EMBEDDING_DEPTH,out_features=EMBEDDING_DEPTH)


    def forward(self,tensor):
        q = rearrange(self.q(tensor),'b n (h d) -> b h n d',b=BATCH_SIZE,h = ATTENTION_HEAD_NUM)
        k = rearrange(self.k(tensor),'b n (h d) -> b h n d',b=BATCH_SIZE,h = ATTENTION_HEAD_NUM)
        v = rearrange(self.v(tensor),'b n (h d) -> b h n d',b=BATCH_SIZE,h = ATTENTION_HEAD_NUM)
        energy = torch.einsum('b h q d , b n k d -> b h q k',q,k)
        att = F.softmax(energy,dim=-1)/EMBEDDING_DEPTH**(1/2)
        att = self.att_drop(att)
        out = torch.einsum('b h a l, b h l v -> b h a v', att, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        out +=tensor
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
    def forward(self,tensor):
        x = self.mlp(tensor)
        x +=tensor
        return x

class ENCODER_BLOCK(nn.Module):
    def __init__(self):
        super(ENCODER_BLOCK, self).__init__()
        self.mha = Multi_Head_Attention()
        self.mlp = MLP()
    def forward(self,tensor):
        tensor = self.mha(tensor)
        tensor = self.mlp(tensor)
        return tensor
class MLP_HEAD(nn.Module):
    def __init__(self):
        super(MLP_HEAD,self).__init__()
        self.norm = nn.LayerNorm(TOKEN_NUM*EMBEDDING_DEPTH)
        self.linear = nn.Linear(TOKEN_NUM*EMBEDDING_DEPTH,224*224)
    def forward(self,tensor):
        tensor = rearrange(tensor, 'b n d -> b (n d)')
        tensor = self.norm(tensor)
        tensor = self.linear(tensor)
        return tensor
class VIT(nn.Module):
    def __init__(self,encoder_block_num = 8):
        super(VIT,self).__init__()
        self.block_num = encoder_block_num
        self.patch_embedding = Patch_Embadding()
        self.encoder_block = ENCODER_BLOCK()
        self.mlp_head = MLP_HEAD()

    def forward(self,tensor):
        tensor = self.patch_embedding(tensor)
        for i in range(self.block_num):
            tensor = self.encoder_block(tensor)
        tensor = self.mlp_head(tensor)
        return tensor



if __name__ == "__main__":
    dummy = torch.randn((8,1,224,224))
    mdoel = VIT()
    dummy = mdoel(dummy)
    print('for debug')