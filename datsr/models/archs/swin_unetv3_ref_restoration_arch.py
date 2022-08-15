import datsr.models.archs.arch_util as arch_util
import torch
import torch.nn as nn
import torch.nn.functional as F
# from datsr.models.archs.DCNv2.dcn_v2 import DCN_sep_pre_multi_offset_flow_similarity as DynAgg
from datsr.models.archs.dcn_v2 import DCN_sep_pre_multi_offset_flow_similarity as DynAgg
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import pdb


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        # H, W = x_size
        H, W = int(x_size[0]), int(x_size[1])
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        x_size = int(x_size[0]), int(x_size[1])
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            # if self.use_checkpoint:
            #     attn_windows = checkpoint.checkpoint(self.attn, x_windows, self.attn_mask)
            # else:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            # if self.use_checkpoint:
            #     attn_windows = checkpoint.checkpoint(self.attn, x_windows, self.calculate_mask(x_size).to(x.device))
            # else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))


        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                           input_resolution=input_resolution,
                           depth=depth,
                           num_heads=num_heads,
                           window_size=window_size,
                           mlp_ratio=mlp_ratio,
                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                           drop=drop, attn_drop=attn_drop,
                           drop_path=drop_path,
                           norm_layer=norm_layer,
                           downsample=downsample,
                           use_checkpoint=use_checkpoint)

        if resi_connection == '1conv1x1':
            self.conv = nn.Linear(dim, dim)
        else:
            if resi_connection == '1conv':
                self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
            elif resi_connection == '3conv':
                # # save parameters AE313
                self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                          nn.Conv2d(dim//4, dim//4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                          nn.Conv2d(dim//4, dim, 3, 1, 1))
                # # # save parameters AE333
                # self.conv = nn.Sequential(nn.Conv2d(dim, dim//4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                #                           nn.Conv2d(dim//4, dim//4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                #                           nn.Conv2d(dim//4, dim, 3, 1, 1))

            # embedding and unembedding after and before conv
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
                norm_layer=None)

            self.patch_unembed = PatchUnEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
                norm_layer=None)

    def forward(self, x, x_size):
        if hasattr(self, 'patch_embed'):
            return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x
        else:
            return self.conv(self.residual_group(x, x_size)) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim


    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, int(x_size[0]), int(x_size[1]))  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * self.num_feat * 9
        flops = 2*H * 2*W * self.num_feat * 3 * 9
        return flops

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale **2)  * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

class ContentExtractor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.body = arch_util.make_layer(
            arch_util.ResidualBlockNoBN, n_blocks, nf=nf)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.default_init_weights([self.conv_first], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        feat = self.body(feat)

        return feat


class SwinBlock(nn.Module):
    def __init__(self,
                 img_size=40,
                 patch_size=1,
                 embed_dim=180,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 resi_connection='1conv',
                 **kwargs):
        super(SwinBlock, self).__init__()

        self.use_checkpoint = use_checkpoint
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(img_size=img_size,        patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):

        # x: [1, 60, 264, 184]
        x_size = torch.tensor(x.shape[2:4])   # [264, 184]
        x = self.patch_embed(x)               # [1, 48576, 60]
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)                  # [1, 48576, 60]
        for layer in self.layers:
            x = layer(x, x_size)              # [1, 48576, 60]
        x = self.norm(x)  # b seq_len c       # [1, 48576, 60]
        x = self.patch_unembed(x, x_size)     # [1, 60, 264, 184]

        return x


class SwinUnetv3RestorationNet(nn.Module):
    def __init__(self, ngf=64, n_blocks=16, groups=8, embed_dim=64, depths=(8,8), num_heads=(8,8), window_size=8, use_checkpoint=False):
        super(SwinUnetv3RestorationNet, self).__init__()
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=ngf, n_blocks=n_blocks)
        self.dyn_agg_restore = DynamicAggregationRestoration(ngf=ngf, n_blocks=n_blocks, groups=groups,
                                            embed_dim=ngf, depths=depths, num_heads=num_heads,
                                            window_size=window_size, use_checkpoint=use_checkpoint)

        arch_util.srntt_init_weights(self, init_type='normal', init_gain=0.02)
        self.re_init_dcn_offset()

    def re_init_dcn_offset(self):
        self.dyn_agg_restore.down_medium_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.down_medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.down_large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.down_large_dyn_agg.conv_offset_mask.bias.data.zero_()

        self.dyn_agg_restore.up_small_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.up_small_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.up_medium_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.up_medium_dyn_agg.conv_offset_mask.bias.data.zero_()
        self.dyn_agg_restore.up_large_dyn_agg.conv_offset_mask.weight.data.zero_()
        self.dyn_agg_restore.up_large_dyn_agg.conv_offset_mask.bias.data.zero_()


    def forward(self, x, pre_offset_flow_sim, img_ref_feat):
        """
        Args:
            x (Tensor): the input image of SRNTT.
            maps (dict[Tensor]): the swapped feature maps on relu3_1, relu2_1
                and relu1_1. depths of the maps are 256, 128 and 64
                respectively.
        """

        base = F.interpolate(x, None, 4, 'bilinear', False)
        content_feat = self.content_extractor(x)

        upscale_restore = self.dyn_agg_restore(base, content_feat, pre_offset_flow_sim, img_ref_feat)

        return upscale_restore + base


class DynamicAggregationRestoration(nn.Module):

    def __init__(self,
                 ngf=64,
                 n_blocks=16,
                 groups=8,
                 img_size=40,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=64,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False
                 ):
        super(DynamicAggregationRestoration, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.num_layers = len(depths)
        self.embed_dim = ngf
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        self.unet_head = nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1)

        # ---------------------- Down ----------------------

        # dynamic aggregation module for relu1_1 reference feature
        self.down_large_offset_conv1 = nn.Conv2d(ngf + 64*2, 64, 3, 1, 1, bias=True)
        self.down_large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.down_large_dyn_agg = DynAgg(64, 64, 3, stride=1, padding=1, dilation=1,
                                    deformable_groups=groups, extra_offset_mask=True)

        # for large scale
        self.down_head_large = nn.Sequential(
            nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.down_body_large = SwinBlock(img_size=160, embed_dim=ngf,
                                depths=depths, num_heads=num_heads, window_size=window_size, use_checkpoint=use_checkpoint)
        self.down_tail_large = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1)


        # dynamic aggregation module for relu2_1 reference feature
        self.down_medium_offset_conv1 = nn.Conv2d(
            ngf + 128*2, 128, 3, 1, 1, bias=True)
        self.down_medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.down_medium_dyn_agg = DynAgg(128, 128, 3, stride=1, padding=1,dilation=1,
                                    deformable_groups=groups, extra_offset_mask=True)

        # for medium scale restoration
        self.down_head_medium = nn.Sequential(
            nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.down_body_medium = SwinBlock(img_size=80, embed_dim=ngf,
                                depths=depths, num_heads=num_heads, window_size=window_size)
        self.down_tail_medium = nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=1)


        # ---------------------- Up ----------------------
        # dynamic aggregation module for relu3_1 reference feature
        self.up_small_offset_conv1 = nn.Conv2d(
            ngf + 256*2, 256, 3, 1, 1, bias=True)  # concat for diff
        self.up_small_offset_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=True)
        self.up_small_dyn_agg = DynAgg(256, 256, 3, stride=1, padding=1, dilation=1,
                                deformable_groups=groups, extra_offset_mask=True)

        # for small scale restoration
        self.up_head_small = nn.Sequential(
            nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.up_body_small = SwinBlock(img_size=40, embed_dim=ngf,
                                depths=depths, num_heads=num_heads, window_size=window_size)
        self.up_tail_small = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))


        # dynamic aggregation module for relu2_1 reference feature
        self.up_medium_offset_conv1 = nn.Conv2d(
            ngf + 128*2, 128, 3, 1, 1, bias=True)
        self.up_medium_offset_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=True)
        self.up_medium_dyn_agg = DynAgg(128, 128, 3, stride=1, padding=1, dilation=1,
                                deformable_groups=groups, extra_offset_mask=True)

        # for medium scale restoration
        self.up_head_medium = nn.Sequential(
            nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.up_body_medium = SwinBlock(img_size=80, embed_dim=ngf,
                                depths=depths, num_heads=num_heads, window_size=window_size)
        self.up_tail_medium = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2), nn.LeakyReLU(0.1, True))


        # dynamic aggregation module for relu1_1 reference feature
        self.up_large_offset_conv1 = nn.Conv2d(ngf + 64*2, 64, 3, 1, 1, bias=True)
        self.up_large_offset_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.up_large_dyn_agg = DynAgg(64, 64, 3, stride=1, padding=1, dilation=1,
                                deformable_groups=groups, extra_offset_mask=True)

        # for large scale
        self.up_head_large = nn.Sequential(
            nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True))
        self.up_body_large = SwinBlock(img_size=160, embed_dim=ngf,
                                depths=depths, num_heads=num_heads, window_size=window_size, use_checkpoint=use_checkpoint)
        self.up_tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1))

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def flow_warp(self,
                  x,
                  flow,
                  interp_mode='bilinear',
                  padding_mode='zeros',
                  align_corners=True):
        """Warp an image or feature map with optical flow.
        Args:
            x (Tensor): Tensor with size (n, c, h, w).
            flow (Tensor): Tensor with size (n, h, w, 2), normal value.
            interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
            padding_mode (str): 'zeros' or 'border' or 'reflection'.
                Default: 'zeros'.
            align_corners (bool): Before pytorch 1.3, the default value is
                align_corners=True. After pytorch 1.3, the default value is
                align_corners=False. Here, we use the True as default.
        Returns:
            Tensor: Warped image or feature map.
        """

        assert x.size()[-2:] == flow.size()[1:3]
        _, _, h, w = x.size()
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).type_as(x),
            torch.arange(0, w).type_as(x))
        grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
        grid.requires_grad = False

        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x,
                               vgrid_scaled,
                               mode=interp_mode,
                               padding_mode=padding_mode,
                               align_corners=align_corners)

        return output

    def forward(self, base, x, pre_offset_flow_sim, img_ref_feat):

        pre_offset = pre_offset_flow_sim[0]
        pre_flow = pre_offset_flow_sim[1]
        pre_similarity = pre_offset_flow_sim[2]

        pre_relu1_swapped_feat = self.flow_warp(img_ref_feat['relu1_1'], pre_flow['relu1_1'])
        pre_relu2_swapped_feat = self.flow_warp(img_ref_feat['relu2_1'], pre_flow['relu2_1'])
        pre_relu3_swapped_feat = self.flow_warp(img_ref_feat['relu3_1'], pre_flow['relu3_1'])

        # Unet
        x0 = self.unet_head(base)    # [B, 64, 160, 160]

        # -------------- Down ------------------
        # large scale
        down_relu1_offset = torch.cat([x0, pre_relu1_swapped_feat, img_ref_feat['relu1_1']], 1)
        down_relu1_offset = self.lrelu(self.down_large_offset_conv1(down_relu1_offset))
        down_relu1_offset = self.lrelu(self.down_large_offset_conv2(down_relu1_offset))
        down_relu1_swapped_feat = self.lrelu(
            self.down_large_dyn_agg([img_ref_feat['relu1_1'], down_relu1_offset],
                               pre_offset['relu1_1'], pre_similarity['relu1_1']))

        h = torch.cat([x0, down_relu1_swapped_feat], 1)
        h = self.down_head_large(h)
        h = self.down_body_large(h) + x0
        x1 = self.down_tail_large(h)  # [B, 64, 80, 80]

        # medium scale
        down_relu2_offset = torch.cat([x1, pre_relu2_swapped_feat, img_ref_feat['relu2_1']], 1)
        down_relu2_offset = self.lrelu(self.down_medium_offset_conv1(down_relu2_offset))
        down_relu2_offset = self.lrelu(self.down_medium_offset_conv2(down_relu2_offset))
        down_relu2_swapped_feat = self.lrelu(
            self.down_medium_dyn_agg([img_ref_feat['relu2_1'], down_relu2_offset],
                                pre_offset['relu2_1'], pre_similarity['relu2_1']))

        h = torch.cat([x1, down_relu2_swapped_feat], 1)
        h = self.down_head_medium(h)
        h = self.down_body_medium(h) + x1
        x2 = self.down_tail_medium(h)    # [9, 128, 40, 40]

        # -------------- Up ------------------

        # dynamic aggregation for relu3_1 reference feature
        relu3_offset = torch.cat([x2, pre_relu3_swapped_feat, img_ref_feat['relu3_1']], 1)
        relu3_offset = self.lrelu(self.up_small_offset_conv1(relu3_offset))
        relu3_offset = self.lrelu(self.up_small_offset_conv2(relu3_offset))
        relu3_swapped_feat = self.lrelu(
            self.up_small_dyn_agg([img_ref_feat['relu3_1'], relu3_offset], pre_offset['relu3_1'], pre_similarity['relu3_1']))

        # small scale
        h = torch.cat([x2, relu3_swapped_feat], 1)
        h = self.up_head_small(h)
        h = self.up_body_small(h) + x2
        x = self.up_tail_small(h)    # [9, 64, 80, 80]

        # dynamic aggregation for relu2_1 reference feature
        relu2_offset = torch.cat([x, pre_relu2_swapped_feat, img_ref_feat['relu2_1']], 1)
        relu2_offset = self.lrelu(self.up_medium_offset_conv1(relu2_offset))
        relu2_offset = self.lrelu(self.up_medium_offset_conv2(relu2_offset))
        relu2_swapped_feat = self.lrelu(
            self.up_medium_dyn_agg([img_ref_feat['relu2_1'], relu2_offset],
                                pre_offset['relu2_1'], pre_similarity['relu2_1']))
        # medium scale
        h = torch.cat([x+x1, relu2_swapped_feat], 1)
        h = self.up_head_medium(h)
        h = self.up_body_medium(h) + x
        x = self.up_tail_medium(h)   # [9, 64, 160, 160]

        # dynamic aggregation for relu1_1 reference feature
        relu1_offset = torch.cat([x, pre_relu1_swapped_feat, img_ref_feat['relu1_1']], 1)
        relu1_offset = self.lrelu(self.up_large_offset_conv1(relu1_offset))
        relu1_offset = self.lrelu(self.up_large_offset_conv2(relu1_offset))
        relu1_swapped_feat = self.lrelu(
            self.up_large_dyn_agg([img_ref_feat['relu1_1'], relu1_offset],
                               pre_offset['relu1_1'], pre_similarity['relu1_1']))
        # large scale
        h = torch.cat([x+x0, relu1_swapped_feat], 1)
        h = self.up_head_large(h)
        h = self.up_body_large(h) + x
        x = self.up_tail_large(h)

        return x
