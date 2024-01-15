import torch
from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F

PATCH_SIZE = 28 # image is 224,224-> after patch size is  (64,28,28)

BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_PATCH = 64 * BATCH_SIZE
projection_dim = 768
num_head = 4
multi_layer_perceptron = [
    projection_dim*2,
    projection_dim
]
transformer_layers = 12
mlp_head_units = [
    2048,
    4096,
    50176
]
keys= nn.Linear(projection_dim,projection_dim)
queries = nn.Linear(projection_dim,projection_dim)
values =nn.Linear(projection_dim,projection_dim)

class ViT(nn.Sequential):
    def __init__(self,
                in_channels: int=1,
                patch_size : int=PATCH_SIZE,
                emb_size :int=projection_dim,
                depth: int=12,
                n_classes:int =10,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, projection_dim),
        TransformerEncoder(depth, emb_size=projection_dim, **kwargs),
        ClassificationHead(emb_size=emb_size))
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size:int =projection_dim):
        super().__init__(
            Reduce('b n e -> b e', reduction ='mean'),
            nn.LayerNorm(projection_dim),
            nn.Linear(projection_dim, 1024),
            nn.Linear(1024,2048),
            nn.Linear(2048,50176)

        )
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth:int =12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = projection_dim, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # Fuse the queries, keys, values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries, and vlaues in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out
class ResidualAdd(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x+= res
        return x
class FeedForwardBlock(nn.Sequential):
    def __init__(self,emb_size:int, expansion : int=4, drop_p :float = 0.):
        super().__init__(
            nn.Linear(projection_dim, expansion* projection_dim),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion*projection_dim, projection_dim))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = projection_dim,
                 drop_p: float = 0,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion,
                    drop_p=forward_drop_p),
            )
            ))


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, patch_size: int = PATCH_SIZE, emb_size: int = projection_dim, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions

        return x
if __name__ == "__main__":
    p = PatchEmbedding(in_channels=1, patch_size= PATCH_SIZE, emb_size = projection_dim, img_size = 224)
    dummy = p(torch.randn(size=(8,1,224,224)))
    x = torch.randn(8, 1, 224, 224)
    patches_embedded = PatchEmbedding()(x)
    x=TransformerEncoderBlock()(patches_embedded).shape
    x = torch.randn(8, 1, 224, 224)
    model = ViT()
    x = model(x)
    print()

