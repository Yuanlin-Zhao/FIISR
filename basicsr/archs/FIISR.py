
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops.layers.torch import Rearrange
from basicsr.utils.registry import ARCH_REGISTRY

######################
# 基础组件与工具
######################

class LayerNorm(nn.Module):
    """ 支持两种格式的 LayerNorm，红外任务中 channels_first 更高效 """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class StripedConv2d(nn.Module):
    """ 条带卷积：捕获红外图像中的长程结构依赖 """
    def __init__(self, in_ch: int, kernel_size: int, depthwise: bool = True):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(1, kernel_size), padding=(0, self.padding),
                      groups=in_ch if depthwise else 1),
            nn.Conv2d(in_ch, in_ch, kernel_size=(kernel_size, 1), padding=(self.padding, 0),
                      groups=in_ch if depthwise else 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x

######################
# 改进后的 Global Block (SME)
######################

class EnhancedStripedConvFormer(nn.Module):
    """ 改进：增加 3x3 局部感知与通道注意力，适应红外图像对比度低、噪声多的特点 """
    def __init__(self, in_ch: int, kernel_size: int):
        super().__init__()
        self.to_qv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1),
            nn.GELU()
        )
        self.attn_striped = StripedConv2d(in_ch, kernel_size=kernel_size, depthwise=True)
        self.attn_local = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_ch // 4, in_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        q, v = self.to_qv(x).chunk(2, dim=1)
        attn = self.attn_striped(q) + self.attn_local(q)
        x = (attn * v)
        x = x * self.ca(x)
        x = self.proj(x) + shortcut
        return x

class ImprovedGatedFFN(nn.Module):
    """ 改进：引入空洞卷积增加上下文背景，使用 Sigmoid 门控增强训练稳定性 """
    def __init__(self, in_ch, mlp_ratio=2, kernel_size=3):
        super().__init__()
        mlp_ch = int(in_ch * mlp_ratio)
        self.fn_1 = nn.Conv2d(in_ch, mlp_ch, kernel_size=1)
        self.fn_2 = nn.Conv2d(mlp_ch // 2, in_ch, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(mlp_ch // 2, mlp_ch // 2, kernel_size=kernel_size,
                      padding=(kernel_size // 2) * 2, dilation=2, groups=mlp_ch // 2),
            nn.Conv2d(mlp_ch // 2, mlp_ch // 2, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = x * torch.sigmoid(self.gate(gate))
        x = self.fn_2(x)
        return x

class SME(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int = 11):
        super().__init__()
        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = EnhancedStripedConvFormer(in_ch=in_ch, kernel_size=kernel_size)
        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = ImprovedGatedFFN(in_ch, mlp_ratio=2, kernel_size=3)
        self.ls1 = nn.Parameter(1e-6 * torch.ones(in_ch, 1, 1))
        self.ls2 = nn.Parameter(1e-6 * torch.ones(in_ch, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1 * self.block(self.norm_1(x))
        x = x + self.ls2 * self.ffn(self.norm_2(x))
        return x

######################
# Local Blocks (RME & MoE)
######################

class Expert(nn.Module):
    def __init__(self, in_ch: int, low_dim: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, low_dim, kernel_size=1)
        self.conv_2 = nn.Conv2d(in_ch, low_dim, kernel_size=1)
        self.conv_3 = nn.Conv2d(low_dim, in_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = (self.conv_2(k) * x)
        x = self.conv_3(x)
        return x

class Router(nn.Module):
    def __init__(self, in_ch: int, num_experts: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 1):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert

    def forward(self, inputs: torch.Tensor, k: torch.Tensor):
        out_logits = self.gate(inputs)
        weights = F.softmax(out_logits, dim=1).to(inputs.dtype)
        topk_weights, topk_experts = torch.topk(weights, self.num_expert)

        output = inputs.clone()
        if self.training:
            exp_weights = torch.zeros_like(weights)
            exp_weights.scatter_(1, topk_experts, weights.gather(1, topk_experts))
            for i, expert in enumerate(self.experts):
                output += expert(inputs, k) * exp_weights[:, i:i + 1, None, None]
        else:
            # 推理阶段只计算选中的专家
            for i in range(self.num_expert):
                expert_idx = topk_experts[:, i]
                expert_weight = topk_weights[:, i:i + 1, None, None]
                for b_idx in range(inputs.shape[0]):
                    # 确保这里所有的 b_idx 都没有空格
                    output[b_idx:b_idx + 1] += self.experts[expert_idx[b_idx]](inputs[b_idx:b_idx + 1],
                                                                               k[b_idx:b_idx + 1]) * expert_weight[
                                                                                                     b_idx:b_idx + 1]
        return output

class MoEBlock(nn.Module):
    def __init__(self, in_ch: int, num_experts: int, topk: int, use_shuffle: bool = False, lr_space: str = "linear", recursive: int = 2):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.recursive = recursive
        self.conv_1 = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(in_ch, 2 * in_ch, kernel_size=1))
        self.agg_conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, groups=in_ch), nn.GELU())
        self.conv = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch), nn.Conv2d(in_ch, in_ch, kernel_size=1))
        self.conv_2 = nn.Sequential(StripedConv2d(in_ch, kernel_size=3, depthwise=True), nn.GELU())

        grow_func = {"linear": lambda i: i + 2, "exp": lambda i: 2** (i + 1), "double": lambda i: 2 * i + 2}[lr_space]
        self.moe_layer = MoELayer(experts=[Expert(in_ch=in_ch, low_dim=grow_func(i)) for i in range(num_experts)],
                                  gate=Router(in_ch=in_ch, num_experts=num_experts), num_expert=topk)
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x
        for _ in range(self.recursive):
            x = self.agg_conv(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return res + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        if self.use_shuffle:
            x = channel_shuffle(x, groups=2)
        x, k = torch.chunk(x, chunks=2, dim=1)
        x = self.conv_2(x)
        k = self.calibrate(k)
        x = self.moe_layer(x, k)
        x = self.proj(x)
        return x


class RME(nn.Module):
    def __init__(self, in_ch: int, num_experts: int, topk: int, lr_space: str = "linear", recursive: int = 2,
                 use_shuffle: bool = False):
        super().__init__()
        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = MoEBlock(in_ch=in_ch, num_experts=num_experts, topk=topk, use_shuffle=use_shuffle,
                              recursive=recursive, lr_space=lr_space)
        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = ImprovedGatedFFN(in_ch, mlp_ratio=2, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.block(self.norm_1(x)) + x

        x = self.ffn(self.norm_2(x)) + x

        return x


class ResGroup(nn.Module):
    def __init__(self, in_ch: int, num_experts: int, global_kernel_size: int, lr_space: str, topk: int, recursive: int,
                 use_shuffle: bool):
        super().__init__()
        self.local_block = RME(in_ch=in_ch, num_experts=num_experts, use_shuffle=use_shuffle, lr_space=lr_space,
                               topk=topk, recursive=recursive)
        self.global_block = SME(in_ch=in_ch, kernel_size=global_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_block(self.local_block(x))


######################
# 最终架构: FIISR
######################

@ARCH_REGISTRY.register()
class FIISR(nn.Module):
    def __init__(self,
                 scale: int = 4,
                 in_chans: int = 3,
                 num_experts: int = 6,
                 num_layers: int = 6,
                 embedding_dim: int = 64,
                 img_range: float = 1.0,
                 use_shuffle: bool = False,
                 global_kernel_size: int = 11,
                 recursive: int = 2,
                 lr_space: str = "linear",
                 topk: int = 2):
        super().__init__()
        self.scale = scale
        self.img_range = img_range

        rgb_mean = (0.4488, 0.4371, 0.4040) if in_chans == 3 else (0.5,)
        self.mean = torch.Tensor(rgb_mean).view(1, in_chans, 1, 1)

        # 浅层特征提取
        self.conv_1 = nn.Conv2d(in_chans, embedding_dim, kernel_size=3, padding=1)

        # 深层特征提取（核心：Local MoE + Global SME）
        self.body = nn.ModuleList([
            ResGroup(in_ch=embedding_dim, num_experts=num_experts, use_shuffle=use_shuffle,
                     topk=topk, lr_space=lr_space, recursive=recursive,
                     global_kernel_size=global_kernel_size) for _ in range(num_layers)
        ])

        # 上采样重建
        self.norm = LayerNorm(embedding_dim, data_format='channels_first')
        self.conv_2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(embedding_dim, (scale ** 2) * in_chans, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_1(x)
        res = x

        for layer in self.body:
            x = layer(x)

        x = self.norm(x)
        x = self.conv_2(x) + res
        x = self.upsampler(x)

        x = x / self.img_range + self.mean
        return x