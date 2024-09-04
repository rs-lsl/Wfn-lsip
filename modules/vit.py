import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from functools import partial

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        input_format="traditional",
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop_rate = attn_drop_rate

        self.proj = nn.Linear(dim, dim)

        if proj_drop_rate > 0:
            self.proj_drop = nn.Dropout(proj_drop_rate)
        else:
            self.proj_drop = nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_rate)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        mlp_drop_rate=0.0,
        attn_drop_rate=0.0,
        path_drop_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        comm_inp_name="fin",
        comm_hidden_name="fout",
    ):
        super().__init__()

        # if (comm.get_size(comm_inp_name) * comm.get_size(comm_hidden_name)) > 1:
        #     self.attn = DistributedAttention(
        #         dim,
        #         input_format="traditional",
        #         comm_inp_name=comm_inp_name,
        #         comm_hidden_name=comm_hidden_name,
        #         num_heads=num_heads,
        #         qkv_bias=qkv_bias,
        #         attn_drop_rate=attn_drop_rate,
        #         proj_drop_rate=mlp_drop_rate,
        #         norm_layer=norm_layer,
        #     )
        # else:
        self.attn = Attention(
            dim, input_format="traditional", num_heads=num_heads, qkv_bias=qkv_bias, attn_drop_rate=attn_drop_rate, proj_drop_rate=mlp_drop_rate, norm_layer=norm_layer
        )
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        # distribute MLP for model parallelism
        # if (comm.get_size(comm_inp_name) * comm.get_size(comm_hidden_name)) > 1:
        #     self.mlp = DistributedMLP(
        #         in_features=dim,
        #         hidden_features=mlp_hidden_dim,
        #         out_features=dim,
        #         act_layer=act_layer,
        #         drop_rate=mlp_drop_rate,
        #         input_format="traditional",
        #         comm_inp_name=comm_inp_name,
        #         comm_hidden_name=comm_hidden_name,
        #     )
        # else:
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop_rate=mlp_drop_rate, input_format="traditional")

    def forward(self, x):
        # flatten transpose:
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x))

        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        inp_shape=[224, 224],
        patch_size=(16, 16),
        inp_chans=3,
        out_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        mlp_drop_rate=0.0,
        attn_drop_rate=0.0,
        path_drop_rate=0.0,
        norm_layer="layer_norm",
        comm_inp_name="fin",
        comm_hidden_name="fout",
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = inp_shape
        self.out_ch = out_chans
        self.comm_inp_name = comm_inp_name
        self.comm_hidden_name = comm_hidden_name

        self.time_conv = nn.Sequential(
            nn.Conv3d(2, 2, [3,3,3], padding=[1,1,1], padding_mode='circular'),
            nn.SiLU(),
            nn.Conv3d(2, 2, [3,3,3], padding=[1,1,1], padding_mode='circular'))

        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=patch_size, in_chans=inp_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches

        # annotate for distributed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_embed.is_shared_mp = []

        self.pos_drop = nn.Dropout(p=path_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, depth)]  # stochastic depth decay rule

        if norm_layer == "layer_norm":
            norm_layer_handle = nn.LayerNorm
        else:
            raise NotImplementedError(f"Error, normalization layer type {norm_layer} not implemented for ViT.")

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    mlp_drop_rate=mlp_drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    path_drop_rate=dpr[i],
                    norm_layer=norm_layer_handle,
                    comm_inp_name=comm_inp_name,
                    comm_hidden_name=comm_hidden_name,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer_handle(embed_dim)

        self.out_size = self.out_ch * self.patch_size[0] * self.patch_size[1]

        self.head = nn.Linear(embed_dim, self.out_size, bias=False)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x).transpose(1, 2)  # patch linear embedding

        # add positional encoding to each token
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_head(self, x):
        B, _, _ = x.shape  # B x N x embed_dim
        x = x.reshape(B, self.patch_embed.red_img_size[0], self.patch_embed.red_img_size[1], self.embed_dim)
        B, h, w, _ = x.shape
        # print(x.shape)

        # apply head
        x = self.head(x)
        x = x.reshape(shape=(B, h, w, self.patch_size[0], self.patch_size[1], self.out_ch))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(B, self.out_ch, self.img_size[0], -1))

        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, 2, -1, H, W)
        x = self.time_conv(x)
        x = self.prepare_tokens(x.view(B, C, H, W))
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.forward_head(x)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, hidden_dim, act_layer, gain=1.0, input_format="nchw"):
        super(EncoderDecoder, self).__init__()

        encoder_modules = []
        current_dim = input_dim
        for i in range(num_layers):
            # fully connected layer
            if input_format == "nchw":
                encoder_modules.append(nn.Conv2d(current_dim, hidden_dim, 1, bias=True))
            elif input_format == "traditional":
                encoder_modules.append(nn.Linear(current_dim, hidden_dim, bias=True))
            else:
                raise NotImplementedError(f"Error, input format {input_format} not supported.")

            # weight sharing
            encoder_modules[-1].weight.is_shared_mp = ["spatial"]

            # proper initializaiton
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
            if encoder_modules[-1].bias is not None:
                encoder_modules[-1].bias.is_shared_mp = ["spatial"]
                nn.init.constant_(encoder_modules[-1].bias, 0.0)

            encoder_modules.append(nn.SiLU())
            current_dim = hidden_dim

        # final output layer
        if input_format == "nchw":
            encoder_modules.append(nn.Conv2d(current_dim, output_dim, 1, bias=False))
        elif input_format == "traditional":
            encoder_modules.append(nn.Linear(current_dim, output_dim, bias=False))

        # weight sharing
        encoder_modules[-1].weight.is_shared_mp = ["spatial"]

        # proper initializaiton
        scale = math.sqrt(gain / current_dim)
        nn.init.normal_(encoder_modules[-1].weight, mean=0.0, std=scale)
        if encoder_modules[-1].bias is not None:
            encoder_modules[-1].bias.is_shared_mp = ["spatial"]
            nn.init.constant_(encoder_modules[-1].bias, 0.0)

        self.fwd = nn.Sequential(*encoder_modules)

    def forward(self, x):
        return self.fwd(x)

@torch.jit.script
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2d ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.red_img_size = ((img_size[0] // patch_size[0]), (img_size[1] // patch_size[1]))
        num_patches = self.red_img_size[0] * self.red_img_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.proj.weight.is_shared_mp = ["spatial"]
        self.proj.bias.is_shared_mp = ["spatial"]

    def forward(self, x):
        # gather input
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        output_bias=True,
        input_format="nchw",
        drop_rate=0.0,
        drop_type="iid",
        checkpointing=2,
        gain=1.0,
        **kwargs,
    ):
        super(MLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # First fully connected layer
        if input_format == "nchw":
            fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
            fc1.weight.is_shared_mp = ["spatial"]
            fc1.bias.is_shared_mp = ["spatial"]
        elif input_format == "traditional":
            fc1 = nn.Linear(in_features, hidden_features, bias=True)
        else:
            raise NotImplementedError(f"Error, input format {input_format} not supported.")

        # initialize the weights correctly
        scale = math.sqrt(2.0 / in_features)
        nn.init.normal_(fc1.weight, mean=0.0, std=scale)
        nn.init.constant_(fc1.bias, 0.0)

        # activation
        act = act_layer()

        # sanity checks
        if (input_format == "traditional") and (drop_type == "features"):
            raise NotImplementedError(f"Error, traditional input format and feature dropout cannot be selected simultaneously")

        # output layer
        if input_format == "nchw":
            fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
            fc2.weight.is_shared_mp = ["spatial"]
            if output_bias:
                fc2.bias.is_shared_mp = ["spatial"]
        elif input_format == "traditional":
            fc2 = nn.Linear(hidden_features, out_features, bias=output_bias)
        else:
            raise NotImplementedError(f"Error, input format {input_format} not supported.")

        # gain factor for the output determines the scaling of the output init
        scale = math.sqrt(gain / hidden_features)
        nn.init.normal_(fc2.weight, mean=0.0, std=scale)
        if fc2.bias is not None:
            nn.init.constant_(fc2.bias, 0.0)

        if drop_rate > 0.0:
            if drop_type == "iid":
                drop = nn.Dropout(drop_rate)
            elif drop_type == "features":
                drop = nn.Dropout2d(drop_rate)
            else:
                raise NotImplementedError(f"Error, drop_type {drop_type} not supported")
        else:
            drop = nn.Identity()

        # create forward pass
        self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)

    @torch.jit.ignore
    def checkpoint_forward(self, x):
        return checkpoint(self.fwd, x, use_reentrant=False)

    def forward(self, x):
        if self.checkpointing >= 2:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)