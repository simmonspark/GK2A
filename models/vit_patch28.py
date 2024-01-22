import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
torch.autograd.set_detect_anomaly(True)
#predefine
#predefined_param
PATCH_SIZE = 32
EMBEDDING_DEPTH = 400
IMG_SIZE = 224
IMG_CHANNEL = 1
BATCH_SIZE = 32
HEAD_NUM = 8
TOKEN_NUM = (IMG_SIZE//PATCH_SIZE)**2 +1
ENCODER_NUM = 8
class PatchEmbedding(nn.Module):
    def __init__(self):
        super(PatchEmbedding,self).__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=IMG_CHANNEL, out_channels=EMBEDDING_DEPTH, kernel_size=PATCH_SIZE, stride=PATCH_SIZE),
            Rearrange('b c w h -> b (w h) c')
        )
        self.cls_token = nn.Parameter(torch.randn(size=(1,1,EMBEDDING_DEPTH)))
        self.position_token = nn.Parameter(torch.rand(size=((IMG_SIZE//PATCH_SIZE)**2+1,EMBEDDING_DEPTH)))

    def forward(self,tensor):
        tensor = self.projection(tensor)
        cls_token = repeat(self.cls_token,'() l d -> b l d',b=tensor.shape[0])
        tensor = torch.cat([tensor,cls_token],dim=1)
        tensor += self.position_token
        return tensor
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
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.MultiHeadAttention = nn.MultiheadAttention(embed_dim=EMBEDDING_DEPTH,num_heads=8,dropout=0.5)
        self.layer = nn.Linear(in_features=EMBEDDING_DEPTH,out_features=EMBEDDING_DEPTH)
        self.mlp = MLP()
    def forward(self,tensor):
        tensor = self.layer(tensor)
        qkv = tensor.clone()
        attention_output, _ = self.MultiHeadAttention(qkv,qkv,qkv)
        tensor +=attention_output
        tensor = self.mlp(tensor)
        return tensor
class VIT(nn.Module):
    def __init__(self):
        super(VIT,self).__init__()
        self.emb = PatchEmbedding()
        self.encoder = Encoder()
        self.mlp_head = MLP_HEAD()
    def forward(self,tensor):
        tensor = self.emb(tensor)
        for i in range(ENCODER_NUM):
            tensor = self.encoder(tensor)
        return rearrange(self.mlp_head(tensor),'b (c w h) -> b c w h',b=tensor.shape[0],w=IMG_SIZE,c=1)



if __name__ == "__main__":
    dummy = torch.randn(size=(BATCH_SIZE,1,224,224))
    model = VIT()
    pred = model(dummy)
    print('for debug')