# import torch.nn as nn
# from torch.nn.init import constant_


# try:
#     from model.ops_dcnv3.functions import DCNv3Function
# except:
#     from model.ops_dcnv3.functions import DCNv3Function
# try:
#     from model.utils import selective_scan_state_flop_jit, selective_scan_fn
# except:
#     from model.utils import selective_scan_state_flop_jit, selective_scan_fn


# class to_channels_first(nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x.permute(0, 3, 1, 2)


# class to_channels_last(nn.Module):

#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return x.permute(0, 2, 3, 1)


# def build_norm_layer(dim,
#                      norm_layer,
#                      in_format='channels_last',
#                      out_format='channels_last',
#                      eps=1e-6):
#     layers = []
#     if norm_layer == 'BN':
#         if in_format == 'channels_last':
#             layers.append(to_channels_first())
#         layers.append(nn.BatchNorm2d(dim))
#         if out_format == 'channels_last':
#             layers.append(to_channels_last())
#     elif norm_layer == 'LN':
#         if in_format == 'channels_first':
#             layers.append(to_channels_last())
#         layers.append(nn.LayerNorm(dim, eps=eps))
#         if out_format == 'channels_first':
#             layers.append(to_channels_first())
#     else:
#         raise NotImplementedError(
#             f'build_norm_layer does not support {norm_layer}')
#     return nn.Sequential(*layers)


# def build_act_layer(act_layer):
#     if act_layer == 'ReLU':
#         return nn.ReLU(inplace=True)
#     elif act_layer == 'SiLU':
#         return nn.SiLU(inplace=True)
#     elif act_layer == 'GELU':
#         return nn.GELU()

#     raise NotImplementedError(f'build_act_layer does not support {act_layer}')




# class CenterFeatureScaleModule(nn.Module):
#     def forward(self,
#                 query,
#                 center_feature_scale_proj_weight,
#                 center_feature_scale_proj_bias):
#         center_feature_scale = F.linear(query,
#                                         weight=center_feature_scale_proj_weight,
#                                         bias=center_feature_scale_proj_bias).sigmoid()
#         return center_feature_scale


# class SpatialAdaptiveScan(nn.Module):
#     def __init__(
#             self,
#             channels=64,
#             kernel_size=1,
#             dw_kernel_size=3,
#             stride=1,
#             pad=0,
#             dilation=1,
#             group=1,
#             offset_scale=0.1,
#             act_layer='GELU',
#             norm_layer='LN',
#             center_feature_scale=False,
#             remove_center=False,
#     ):
#         super().__init__()
#         if channels % group != 0:
#             raise ValueError(
#                 f'channels must be divisible by group, but got {channels} and {group}')
#         _d_per_group = channels // group
#         dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size


#         self.offset_scale = offset_scale
#         self.channels = channels
#         self.kernel_size = kernel_size
#         self.dw_kernel_size = dw_kernel_size
#         self.stride = stride
#         self.dilation = dilation
#         self.pad = pad
#         self.group = group
#         self.group_channels = channels // group
#         self.offset_scale = offset_scale
#         self.center_feature_scale = center_feature_scale
#         self.remove_center = int(remove_center)

#         if self.remove_center and self.kernel_size % 2 == 0:
#             raise ValueError('remove_center is only compatible with odd kernel size.')

#         self.dw_conv = nn.Sequential(
#             nn.Conv2d(
#                 channels,
#                 channels,
#                 kernel_size=dw_kernel_size,
#                 stride=1,
#                 padding=(dw_kernel_size - 1) // 2,
#                 groups=channels),
#             build_norm_layer(
#                 channels,
#                 norm_layer,
#                 'channels_first',
#                 'channels_last'),
#             build_act_layer(act_layer))
#         self.offset = nn.Linear(
#             channels,
#             group * (kernel_size * kernel_size - remove_center) * 2)
#         self._reset_parameters()

#         if center_feature_scale:
#             self.center_feature_scale_proj_weight = nn.Parameter(
#                 torch.zeros((group, channels), dtype=torch.float))
#             self.center_feature_scale_proj_bias = nn.Parameter(
#                 torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
#             self.center_feature_scale_module = CenterFeatureScaleModule()

#     def _reset_parameters(self):
#         constant_(self.offset.weight.data, 0.)
#         constant_(self.offset.bias.data, 0.)


#     def forward(self, input, x):
#         N, _, H, W = input.shape
#         x_proj = x
#         x1 = input
#         x1 = self.dw_conv(x1)
#         offset = self.offset(x1)
#         mask = torch.ones(N, H, W, self.group, device=x.device, dtype=x.dtype)
#         x = DCNv3Function.apply(
#             x, offset, mask,
#             self.kernel_size, self.kernel_size,
#             self.stride, self.stride,
#             self.pad, self.pad,
#             self.dilation, self.dilation,
#             self.group, self.group_channels,
#             self.offset_scale,
#             256,
#             self.remove_center)

#         if self.center_feature_scale:
#             center_feature_scale = self.center_feature_scale_module(
#                 x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
#             # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
#             center_feature_scale = center_feature_scale[..., None].repeat(
#                 1, 1, 1, 1, self.channels // self.group).flatten(-2)
#             x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

#         x = x.permute(0, 3, 1, 2).contiguous()
#         return x


# class SpectralAdaptiveScan(nn.Module):
#     """
#     光谱自适应扫描模块 - 针对高光谱数据的光谱维度自适应扫描
#     """

#     def __init__(
#             self,
#             channels,  # 光谱通道数
#             kernel_size=3,  # 1D卷积核大小
#             stride=1,
#             pad=1,
#             group=1,
#             # offset_scale=1.0,
#             offset_scale=1.0,
#             act_layer='GELU',
#             norm_layer='LN',
#             spectral_reduction=False,  # 是否进行光谱降维
#             reduction_ratio=2,  # 降维比例
#     ):
#         super().__init__()

#         self.channels = channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.pad = pad
#         self.group = group
#         self.offset_scale = offset_scale
#         self.spectral_reduction = spectral_reduction
#         self.reduction_ratio = reduction_ratio

#         # 如果启用光谱降维
#         if spectral_reduction:
#             self.reduced_channels = channels // reduction_ratio
#         else:
#             self.reduced_channels = channels

#         # 1D深度卷积用于光谱特征提取
#         self.spectral_conv = nn.Sequential(
#             nn.Conv1d(
#                 channels,
#                 channels,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 padding=pad,
#                 groups=channels  # 深度卷积
#             ),
#             self._build_norm_layer(channels, norm_layer),
#             self._build_act_layer(act_layer)
#         )

#         # 生成光谱偏移的线性层
#         self.offset_generator = nn.Linear(
#             channels,
#             group * kernel_size  # 每个group生成kernel_size个偏移
#         )

#         # 光谱权重生成器
#         self.weight_generator = nn.Linear(
#             channels,
#             group * kernel_size
#         )

#         # 可选的光谱降维层
#         if spectral_reduction:
#             self.spectral_reduction_layer = nn.Conv1d(
#                 channels,
#                 self.reduced_channels,
#                 kernel_size=1
#             )
#             self.spectral_expansion_layer = nn.Conv1d(
#                 self.reduced_channels,
#                 channels,
#                 kernel_size=1
#             )

#         self._reset_parameters()

#     def _build_norm_layer(self, dim, norm_layer):
#         if norm_layer == 'BN':
#             return nn.BatchNorm1d(dim)
#         elif norm_layer == 'LN':
#             # 对于1D卷积，我们需要使用BatchNorm1d而不是LayerNorm
#             # 因为1D卷积的输出格式是[B, C, L]，而LayerNorm期待[B, L, C]
#             return nn.BatchNorm1d(dim)
#         else:
#             return nn.Identity()

#     def _build_act_layer(self, act_layer):
#         if act_layer == 'ReLU':
#             return nn.ReLU(inplace=True)
#         elif act_layer == 'SiLU':
#             return nn.SiLU(inplace=True)
#         elif act_layer == 'GELU':
#             return nn.GELU()
#         else:
#             return nn.Identity()

#     def _reset_parameters(self):
#         from torch.nn.init import constant_
#         constant_(self.offset_generator.weight.data, 0.)
#         constant_(self.offset_generator.bias.data, 0.)
#         constant_(self.weight_generator.weight.data, 0.)
#         constant_(self.weight_generator.bias.data, 0.)

#     def spectral_adaptive_sample(self, x, offsets, weights):
#         """
#         执行光谱自适应采样
#         Args:
#             x: [B, C, H, W] 或 [B, C, L] 光谱特征
#             offsets: [B, H, W, G*K] 或 [B, L, G*K] 光谱偏移
#             weights: [B, H, W, G*K] 或 [B, L, G*K] 采样权重
#         """
#         if len(x.shape) == 4:  # [B, C, H, W]
#             B, C, H, W = x.shape
#             # 重塑为 [B, H*W, C] 便于处理
#             x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
#             offsets = offsets.view(B, H * W, -1)
#             weights = weights.view(B, H * W, -1)

#             # 执行光谱采样
#             sampled = self._adaptive_spectral_sample_1d(x_reshaped, offsets, weights)

#             # 重塑回原始形状
#             sampled = sampled.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

#         else:  # [B, C, L]
#             sampled = self._adaptive_spectral_sample_1d(
#                 x.permute(0, 2, 1), offsets, weights
#             ).permute(0, 2, 1)

#         return sampled

#     def _adaptive_spectral_sample_1d(self, x, offsets, weights):
#         """
#         1D光谱自适应采样的核心实现
#         Args:
#             x: [B, L, C] 输入特征
#             offsets: [B, L, G*K] 光谱偏移
#             weights: [B, L, G*K] 采样权重
#         """
#         B, L, C = x.shape
#         G = self.group
#         K = self.kernel_size

#         # 重塑偏移和权重
#         offsets = offsets.view(B, L, G, K)
#         weights = weights.view(B, L, G, K)
#         weights = F.softmax(weights, dim=-1)  # 归一化权重

#         # 为每个group生成采样结果
#         group_size = C // G
#         outputs = []

#         for g in range(G):
#             start_idx = g * group_size
#             end_idx = (g + 1) * group_size if g < G - 1 else C
#             x_group = x[:, :, start_idx:end_idx]  # [B, L, group_size]

#             offset_group = offsets[:, :, g, :]  # [B, L, K]
#             weight_group = weights[:, :, g, :]  # [B, L, K]

#             # 为当前group执行自适应采样
#             sampled_group = self._sample_spectral_neighbors(
#                 x_group, offset_group, weight_group
#             )
#             outputs.append(sampled_group)

#         return torch.cat(outputs, dim=-1)

#     def _sample_spectral_neighbors(self, x, offsets, weights):
#         B, L, group_size = x.shape
#         K = self.kernel_size

#         base_indices = torch.arange(group_size, device=x.device, dtype=torch.float32)
#         base_indices = base_indices.view(1, 1, 1, -1).expand(B, L, K, -1)

#         offsets_expanded = offsets.unsqueeze(-1)  # [B, L, K, 1]
#         sample_indices = base_indices + offsets_expanded  # [B, L, K, group_size]

#         sample_indices = torch.clamp(sample_indices, 0, group_size - 1)
#         sampled_features = []
#         for k in range(K):
#             indices = sample_indices[:, :, k, :]  # [B, L, group_size]

#             indices_floor = torch.floor(indices).long()
#             indices_ceil = torch.ceil(indices).long()
#             indices_frac = indices - indices_floor.float()

#             indices_floor = torch.clamp(indices_floor, 0, group_size - 1)
#             indices_ceil = torch.clamp(indices_ceil, 0, group_size - 1)

#             x_floor = torch.gather(x, 2, indices_floor)
#             x_ceil = torch.gather(x, 2, indices_ceil)
#             x_interp = x_floor * (1 - indices_frac) + x_ceil * indices_frac

#             sampled_features.append(x_interp)

#         sampled_features = torch.stack(sampled_features, dim=2)  # [B, L, K, group_size]
#         weights_expanded = weights.unsqueeze(-1)  # [B, L, K, 1]

#         output = torch.sum(sampled_features * weights_expanded, dim=2)  # [B, L, group_size]

#         return output

#     def forward(self, input_feat, x):
#         if len(x.shape) == 4:
#             B, C, H, W = x.shape
#             x_for_conv = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C).unsqueeze(-1)
#             x_for_conv = x_for_conv.permute(0, 2, 1)

#             spectral_feat = self.spectral_conv(x_for_conv)
#             spectral_feat = spectral_feat.squeeze(-1)

#             offsets = self.offset_generator(spectral_feat)
#             weights = self.weight_generator(spectral_feat)

#             offsets = offsets.view(B, H, W, -1)
#             weights = weights.view(B, H, W, -1)

#         else:
#             B, C, L = x.shape
#             spectral_feat = self.spectral_conv(x)
#             spectral_feat = spectral_feat.permute(0, 2, 1)

#             offsets = self.offset_generator(spectral_feat)
#             weights = self.weight_generator(spectral_feat)

#         output = self.spectral_adaptive_sample(x, offsets, weights)
#         if self.spectral_reduction:
#             if len(output.shape) == 4:
#                 B, C, H, W = output.shape
#                 output_flat = output.permute(0, 2, 3, 1).contiguous().view(-1, C).unsqueeze(-1)
#                 output_flat = output_flat.permute(0, 2, 1)

#                 reduced = self.spectral_reduction_layer(output_flat)
#                 expanded = self.spectral_expansion_layer(reduced)

#                 output = expanded.squeeze(-1).view(B, H, W, C).permute(0, 3, 1, 2)
#             else:
#                 reduced = self.spectral_reduction_layer(output)
#                 output = self.spectral_expansion_layer(reduced)

#         return output



# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# from typing import Optional, Callable
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
# from einops import rearrange, repeat

# def random_seed_setting(seed: int = 42):
#     import random
#     import os
#     import numpy as np
#     import torch

#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed)
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)

# class MS2D(nn.Module):
#     def __init__(
#             self,
#             d_model,
#             d_state=16,
#             d_conv=3,
#             expand=1.,
#             dt_rank="auto",
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             dropout=0.,
#             conv_bias=True,
#             bias=False,
#             device=None,
#             dtype=None,
#             **kwargs,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

#         self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
#         self.conv2d = nn.Conv2d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             groups=self.d_inner,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#             **factory_kwargs,
#         )

#         self.DWconv1 = nn.Conv2d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             groups=self.d_inner,
#             bias=conv_bias,
#             kernel_size=3,
#             padding=(3 - 1) // 2,
#             **factory_kwargs,
#         )

#         self.DWconv2 = nn.Conv2d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             groups=self.d_inner,
#             bias=conv_bias,
#             kernel_size=5,
#             padding=(5 - 1) // 2,
#             stride=2,
#             **factory_kwargs,
#         )

#         self.conv_transpose = nn.ConvTranspose2d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             kernel_size=5,
#             stride=2,
#             padding=(5 - 1) // 2,
#         )
#         self.act = nn.SiLU()

#         self.x_proj = (
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#         )
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
#         del self.x_proj

#         self.dt_projs = (
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#         )
#         self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
#         self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
#         del self.dt_projs

#         self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=False)
#         self.Ds = self.D_init(self.d_inner, copies=4, merge=False)

#         self.selective_scan = selective_scan_fn

#         self.out_norm = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else None

#         self.da_scan = SpatialAdaptiveScan(channels=self.d_inner, group=1)
#         random_seed_setting(6)


#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
#                 **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError

#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)

#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)

#         dt_proj.bias._no_reinit = True

#         return dt_proj

#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):

#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)
#         if copies > 1:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log

#     @staticmethod
#     def D_init(d_inner, copies=1, device=None, merge=False):

#         D = torch.ones(d_inner, device=device)
#         if copies > 1:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)
#         D._no_weight_decay = True
#         return D
#     def zigzag_scan(self, x):
#         """
#         之字形扫描函数
#         input: x [B, C, H, W]
#         output: [B, C, H*W] 按之字形顺序排列
#         """
#         B, C, H, W = x.shape
#         device = x.device

#         # 生成之字形扫描的索引
#         indices = []

#         k = 0
#         i = 0
#         j = 0

#         while i < H and j < W and k < H * W:
#             # 将当前位置的线性索引添加到列表中
#             indices.append(i * W + j)
#             k += 1

#             # i + j 为偶数，右上移动
#             if (i + j) % 2 == 0:
#                 # 右边界超出，则向下
#                 if (i - 1) >= 0 and (j + 1) >= W:
#                     i = i + 1
#                 # 上边界超出，则向右
#                 elif (i - 1) < 0 and (j + 1) < W:
#                     j = j + 1
#                 # 上右边界都超出，即处于右上顶点的位置，则向下
#                 elif (i - 1) < 0 and (j + 1) >= W:
#                     i = i + 1
#                 else:
#                     i = i - 1
#                     j = j + 1
#             # i + j 为奇数，左下移动
#             elif (i + j) % 2 == 1:
#                 # 左边界超出，则向下
#                 if (i + 1) < H and (j - 1) < 0:
#                     i = i + 1
#                 # 下边界超出，则向右
#                 elif (i + 1) >= H and (j - 1) >= 0:
#                     j = j + 1
#                 # 左下边界都超出，即处于左下顶点的位置，则向右
#                 elif (i + 1) >= H and (j - 1) < 0:
#                     j = j + 1
#                 else:
#                     i = i + 1
#                     j = j - 1

#         # 转换为张量索引
#         indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)

#         # 将特征图按之字形顺序重新排列
#         x_flat = x.view(B, C, H * W)
#         x_zigzag = torch.gather(x_flat, dim=2, index=indices_tensor.unsqueeze(0).unsqueeze(0).expand(B, C, -1))

#         return x_zigzag

#     def forward_core(self, x: torch.Tensor):
#         """
#         简化的之字形扫描forward_core
#         input: x [B, C, H, W]
#         output: [B, C, H*W] 按之字形顺序排列
#         """
#         # 直接对输入进行之字形扫描
#         y = self.zigzag_scan(x)

#         return y

#     def forward(self, x: torch.Tensor, **kwargs):
#         B, H, W, C = x.shape
#         input_ = x.permute(0, 3, 1, 2).contiguous()

#         xz = self.in_proj(x)
#         x, z = xz.chunk(2, dim=-1)

#         x = x.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
#         x = self.act(self.conv2d(x))

#         x = self.da_scan(input_, x.permute(0, 2, 3, 1).contiguous())

#         y1 = self.forward_core(x)
#         assert y1.dtype == torch.float32
#         # y = y1 + y2 + y3 + y4
#         y = y1
#         y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
#         y = self.out_norm(y)
#         y = y * F.silu(z)
#         out = self.out_proj(y)
#         if self.dropout is not None:
#             out = self.dropout(out)
#         # print(out.shape)
#         # print(self.d_model)
#         return out


# class Fuse_SS2D(nn.Module):
#     def __init__(
#             self,
#             d_model1,
#             d_model2,
#             d_state=16,
#             expand=2.,
#             dt_rank="auto",
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             dropout=0.,
#             bias=False,
#             device=None,
#             dtype=None,
#             **kwargs,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model1 = d_model1
#         self.d_model2 = d_model2
#         self.d_state = d_state
#         self.expand = expand
#         self.d_inner1 = int(self.expand * self.d_model1)
#         self.d_inner2 = int(self.expand * self.d_model2)

#         self.dt_rank = math.ceil(self.d_model1 / 16) if dt_rank == "auto" else dt_rank

#         self.x_proj = (
#             nn.Linear(self.d_inner1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#         )
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
#         del self.x_proj

#         self.dt_projs = (
#             self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#         )
#         self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
#         self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
#         del self.dt_projs

#         self.A_logs = self.A_log_init(self.d_state, self.d_inner2, copies=4, merge=True)  # (K=4, D, N)
#         self.Ds = self.D_init(self.d_inner2, copies=4, merge=True)  # (K=4, D, N)

#         self.selective_scan = selective_scan_fn

#         self.out_norm = nn.LayerNorm(self.d_inner2)

#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
#                 **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError

#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)
#         dt_proj.bias._no_reinit = True

#         return dt_proj

#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)  # Keep A_log in fp32
#         if copies > 1:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log

#     @staticmethod
#     def D_init(d_inner, copies=1, device=None, merge=True):
#         # D "skip" parameter
#         D = torch.ones(d_inner, device=device)
#         if copies > 1:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)  # Keep in fp32
#         D._no_weight_decay = True
#         return D

#     def forward_core(self, x, y):
#         B, C, H, W = x.shape
#         L = H * W
#         K = 4
#         x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
#                              dim=1).view(B, 2, -1, L)
#         xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)

#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#         xs = xs.float().view(B, -1, L)
#         dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
#         Bs = Bs.float().view(B, K, -1, L)
#         Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
#         Ds = self.Ds.float().view(-1)
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
#         dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

#         y_hwwh = torch.stack([y.view(B, -1, L), torch.transpose(y, dim0=2, dim1=3).contiguous().view(B, -1, L)],
#                              dim=1).view(B, 2, -1, L)
#         ys = torch.cat([y_hwwh, torch.flip(y_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
#         ys = ys.float().view(B, -1, L)

#         out_y = self.selective_scan(
#             ys, dts,
#             As, Bs, Cs, Ds, z=None,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#             return_last_state=False,
#         ).view(B, K, -1, L)
#         assert out_y.dtype == torch.float

#         inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
#         wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
#         invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

#         return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

#     def forward(self, x, y):
#         B, C, H, W = x.shape

#         ya1, ya2, ya3, ya4 = self.forward_core(x, y)

#         ya = ya1 + ya2 + ya3 + ya4
#         ya = torch.transpose(ya, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
#         ya = self.out_norm(ya)
#         return ya


# class FSSBlock(nn.Module):
#     def __init__(
#             self,
#             hidden_dim1: int = 0,
#             hidden_dim2: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             expand: float = 0.5,
#             d_conv=3,
#             bias=False,
#             conv_bias=True,
#             **kwargs,
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(hidden_dim1)
#         self.ln_2 = norm_layer(hidden_dim2)
#         self.expand = expand
#         self.d_inner1 = int(self.expand * hidden_dim1)
#         self.d_inner2 = int(self.expand * hidden_dim2)

#         self.in_proj1 = nn.Linear(hidden_dim1, self.d_inner1 * 2, bias=bias)
#         self.in_proj2 = nn.Linear(hidden_dim2, self.d_inner2 * 2, bias=bias)

#         self.conv2d1 = nn.Conv2d(
#             in_channels=self.d_inner1,
#             out_channels=self.d_inner1,
#             groups=self.d_inner1,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#         )
#         self.conv2d2 = nn.Conv2d(
#             in_channels=self.d_inner2,
#             out_channels=self.d_inner2,
#             groups=self.d_inner2,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#         )
#         self.act = nn.SiLU()

#         self.attention1 = Fuse_SS2D(d_model1=hidden_dim1, d_model2=hidden_dim2, d_state=d_state, expand=expand,
#                                     dropout=attn_drop_rate, **kwargs)  # 代码中的SS2D完成的内容更多
#         self.attention2 = Fuse_SS2D(d_model1=hidden_dim2, d_model2=hidden_dim1, d_state=d_state, expand=expand,
#                                     dropout=attn_drop_rate, **kwargs)
#         self.out_proj1 = nn.Linear(self.d_inner1, hidden_dim1, bias=bias)
#         self.out_proj2 = nn.Linear(self.d_inner2, hidden_dim2, bias=bias)
#         self.drop_path = DropPath(drop_path)

#     def forward(self, x, y):
#         dim_num = len(x.size())
#         B, C, N, H, W = 0, 0, 0, 0, 0
#         if (dim_num == 5):
#             B, C, N, H, W = x.size()
#             x = x.reshape(B, C * N, H, W)
#         else:
#             B, C, H, W = x.size()

#         x = x.reshape(B, H, W, -1)
#         y = y.reshape(B, H, W, -1)

#         x_ = self.ln_1(x)
#         y_ = self.ln_2(y)

#         x_12 = self.in_proj1(x_)
#         x_1, x_2 = x_12.chunk(2, dim=-1)

#         y_12 = self.in_proj2(y_)
#         y_1, y_2 = y_12.chunk(2, dim=-1)

#         x_1 = x_1.reshape(B, -1, H, W)
#         y_1 = y_1.reshape(B, -1, H, W)

#         x_1 = self.act(self.conv2d1(x_1))
#         y_1 = self.act(self.conv2d2(y_1))

#         y_out = self.attention1(x_1, y_1)
#         x_out = self.attention2(y_1, x_1)

#         x_out = x_out * F.silu(x_2)
#         y_out = y_out * F.silu(y_2)

#         out_x = self.out_proj1(x_out)
#         out_y = self.out_proj2(y_out)

#         x = x + out_x
#         y = y + out_y

#         x = x.reshape(B, -1, H, W)
#         y = y.reshape(B, -1, H, W)
#         if (dim_num == 5):
#             x = x.reshape(B, C, N, H, W)
#         return x, y


# class MSBlock(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             expand: float = 0.5,
#             **kwargs,
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = MS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
#         self.drop_path = DropPath(drop_path)

#     def forward(self, input):
#         cate = 'x'
#         B, C, N, H, W = 0, 0, 0, 0, 0

#         if (len(input.size()) == 5):
#             B, C, N, H, W = input.size()
#             input = input.reshape(B, C * N, H, W)
#             cate = 'h'
#         else:
#             B, C, H, W = input.size()
#         input = input.reshape(B, H, W, -1)
#         x = self.ln_1(input)
#         x = self.self_attention(x)
#         x = input + x
#         x = x.permute(0, 3, 1, 2).contiguous()
#         if (cate == 'h'):
#             x = x.reshape(B, C, N, H, W)
#         return x


# class Spec_SS1D(nn.Module):
#     def __init__(
#             self,
#             d_model,
#             d_state=16,
#             d_conv=3,
#             expand=1.,
#             dt_rank="auto",
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             dropout=0.,
#             conv_bias=True,
#             bias=False,
#             device=None,
#             dtype=None,
#             **kwargs,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

#         self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
#         self.proj = nn.Linear(self.d_inner, self.d_inner, bias=bias)
#         self.act = nn.SiLU()

#         self.x_proj = (
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#         )
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
#         del self.x_proj

#         self.dt_projs = (
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#         )
#         self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
#         self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
#         del self.dt_projs

#         self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
#         self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

#         self.selective_scan = selective_scan_fn

#         self.out_norm = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else None

#         self.da_scan = SpatialAdaptiveScan(channels=self.d_inner, group=1)

#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
#                 **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

#         # Initialize special dt projection to preserve variance at initialization
#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError

#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)
#         dt_proj.bias._no_reinit = True

#         return dt_proj

#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
#         # S4D real initialization
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)  # Keep A_log in fp32
#         if copies > 1:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log

#     @staticmethod
#     def D_init(d_inner, copies=1, device=None, merge=True):
#         # D "skip" parameter
#         D = torch.ones(d_inner, device=device)
#         if copies > 1:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)  # Keep in fp32
#         D._no_weight_decay = True
#         return D

#     def forward_core(self, x: torch.Tensor):
#         B, C, D = x.shape
#         L = C
#         K = 2
#         xs = torch.stack([x.view(B, -1, L), torch.flip(x.view(B, -1, L), dims=[-1])], dim=1)

#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#         xs = xs.float().view(B, -1, L)
#         dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
#         Bs = Bs.float().view(B, K, -1, L)
#         Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
#         Ds = self.Ds.float().view(-1)
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
#         dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
#         out_y = self.selective_scan(
#             xs, dts,
#             As, Bs, Cs, Ds, z=None,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#             return_last_state=False,
#         ).view(B, K, -1, L)
#         # print(out_y.shape)
#         assert out_y.dtype == torch.float
#         y2 = torch.flip(out_y[:, 1], dims=[-1])
#         return out_y[:, 0], y2

#     def forward(self, x: torch.Tensor, **kwargs):
#         # input_ = x.contiguous()
#         xz = self.in_proj(x)
#         x, z = xz.chunk(2, dim=-1)

#         x = self.act(self.proj(x))

#         # x = self.da_scan(input_, x.contiguous())

#         y1, y2 = self.forward_core(x)
#         assert y1.dtype == torch.float32
#         y = y1 + y2
#         y = torch.transpose(y, dim0=1, dim1=2).contiguous()
#         y = self.out_norm(y)
#         y = y * F.silu(z)
#         out = self.out_proj(y)
#         if self.dropout is not None:
#             out = self.dropout(out)
#         return out


# class SpecMambaBlock(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             expand: float = 0.5,
#             is_light_sr: bool = False,
#             **kwargs,
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = Spec_SS1D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate,
#                                         **kwargs)
#         self.drop_path = DropPath(drop_path)

#     def forward(self, input):
#         B, C, N, H, W = input.size()
#         input = input.reshape(B, C * N, H * W)

#         x = self.ln_1(input)
#         x = self.self_attention(x)
#         x = x + input
#         x = x.reshape(B, C, N, H, W)
#         return x


# class SpectralAdaptive_SS1D(nn.Module):
#     """
#     结合光谱自适应扫描的1D状态空间模型
#     """

#     def __init__(
#             self,
#             d_model,
#             d_state=16,
#             d_conv=3,
#             expand=1.,
#             dt_rank="auto",
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             dropout=0.,
#             conv_bias=True,
#             bias=False,
#             device=None,
#             dtype=None,
#             spectral_adaptive=True,  # 是否使用光谱自适应扫描
#             **kwargs,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
#         self.spectral_adaptive = spectral_adaptive

#         self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
#         self.proj = nn.Linear(self.d_inner, self.d_inner, bias=bias)
#         self.act = nn.SiLU()

#         self.x_proj = (
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
#         )
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
#         del self.x_proj

#         self.dt_projs = (
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
#                          **factory_kwargs),
#         )
#         self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
#         self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
#         del self.dt_projs

#         self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)
#         self.Ds = self.D_init(self.d_inner, copies=2, merge=True)

#         self.selective_scan = selective_scan_fn

#         self.out_norm = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else None

#         # 光谱自适应扫描模块
#         if self.spectral_adaptive:
#             self.spectral_scan = SpectralAdaptiveScan(
#                 channels=self.d_inner,
#                 group=4,  # 可以调整group数量
#                 **kwargs
#             )

#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
#                 **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError

#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)
#         dt_proj.bias._no_reinit = True

#         return dt_proj

#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)
#         if copies > 1:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log

#     @staticmethod
#     def D_init(d_inner, copies=1, device=None, merge=True):
#         D = torch.ones(d_inner, device=device)
#         if copies > 1:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)
#         D._no_weight_decay = True
#         return D

#     def forward_core(self, x: torch.Tensor):
#         B, C, D = x.shape
#         L = C
#         K = 2
#         xs = torch.stack([x.view(B, -1, L), torch.flip(x.view(B, -1, L), dims=[-1])], dim=1)

#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
#         xs = xs.float().view(B, -1, L)
#         dts = dts.contiguous().float().view(B, -1, L)
#         Bs = Bs.float().view(B, K, -1, L)
#         Cs = Cs.float().view(B, K, -1, L)
#         Ds = self.Ds.float().view(-1)
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
#         dt_projs_bias = self.dt_projs_bias.float().view(-1)

#         out_y = self.selective_scan(
#             xs, dts,
#             As, Bs, Cs, Ds, z=None,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#             return_last_state=False,
#         ).view(B, K, -1, L)

#         assert out_y.dtype == torch.float
#         y2 = torch.flip(out_y[:, 1], dims=[-1])
#         return out_y[:, 0], y2

#     def forward(self, x: torch.Tensor, **kwargs):
#         xz = self.in_proj(x)
#         x_inner, z = xz.chunk(2, dim=-1)

#         x_inner = self.act(self.proj(x_inner))

#         # 如果启用光谱自适应扫描
#         if self.spectral_adaptive:
#             # 转换为适合光谱扫描的格式
#             if len(x_inner.shape) == 3:  # [B, L, D]
#                 x_inner = x_inner.permute(0, 2, 1)  # [B, D, L]

#             x_inner = self.spectral_scan(x_inner, x_inner)

#             if len(x_inner.shape) == 3:  # [B, D, L]
#                 x_inner = x_inner.permute(0, 2, 1)  # [B, L, D]

#         y1, y2 = self.forward_core(x_inner)
#         assert y1.dtype == torch.float32
#         y = y1 + y2
#         y = torch.transpose(y, dim0=1, dim1=2).contiguous()
#         y = self.out_norm(y)
#         y = y * F.silu(z)
#         out = self.out_proj(y)
#         if self.dropout is not None:
#             out = self.dropout(out)
#         return out


# class SimpleSpecMambaBlock(nn.Module):
#     """
#     光谱自适应Mamba块
#     """

#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             norm_layer=nn.LayerNorm,
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             expand: float = 0.5,
#             spectral_adaptive: bool = True,
#             dropout: float = 0.,  # 显式添加dropout参数
#             **kwargs,
#     ):
#         super().__init__()
#         self.ln_1 = norm_layer(hidden_dim)

#         # 从kwargs中移除可能重复的参数
#         clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['dropout', 'attn_drop_rate']}

#         self.self_attention = SpectralAdaptive_SS1D(
#             d_model=hidden_dim,
#             d_state=d_state,
#             expand=expand,
#             dropout=dropout,  # 使用显式传入的dropout参数
#             spectral_adaptive=spectral_adaptive,
#             **clean_kwargs
#         )
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, input):
#         original_shape = input.shape

#         if len(input.shape) == 5:  # [B, C, N, H, W]
#             B, C, N, H, W = input.size()
#             input = input.reshape(B, C * N, H * W)
#             input = input.permute(0, 2, 1)  # [B, H*W, C*N]
#             target_shape = (B, C, N, H, W)
#         elif len(input.shape) == 4:  # [B, C, H, W]
#             B, C, H, W = input.shape
#             input = input.reshape(B, C, H * W)
#             input = input.permute(0, 2, 1)  # [B, H*W, C]
#             target_shape = (B, C, H, W)
#         else:
#             # 已经是正确的3D形状 [B, L, D]
#             target_shape = original_shape

#         x = self.ln_1(input)
#         x = self.self_attention(x)
#         x = input + self.drop_path(x)

#         # 重塑回原始形状
#         if len(original_shape) == 5:
#             B, C, N, H, W = target_shape
#             x = x.permute(0, 2, 1).reshape(B, C * N, H * W).reshape(B, C, N, H, W)
#         elif len(original_shape) == 4:
#             B, C, H, W = target_shape
#             x = x.permute(0, 2, 1).reshape(B, C, H * W).reshape(B, C, H, W)
#         # 如果是3D，保持不变

#         return x




import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import torch.nn as nn
from torch.nn.init import constant_
try:
    from model.ops_dcnv3.functions import DCNv3Function
except:
    from model.ops_dcnv3.functions import DCNv3Function
try:
    from model.utils import selective_scan_state_flop_jit, selective_scan_fn
except:
    from model.utils import selective_scan_state_flop_jit, selective_scan_fn


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)

def random_seed_setting(seed: int = 42):
    import random
    import os
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')




class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class SpatialAdaptiveScan(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=1,
            dw_kernel_size=3,
            stride=1,
            pad=0,
            dilation=1,
            group=1,
            offset_scale=0.1,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
    ):
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size


        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)


    def forward(self, input, x):
        N, _, H, W = input.shape
        x_proj = x
        x1 = input
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = torch.ones(N, H, W, self.group, device=x.device, dtype=x.dtype)
        x = DCNv3Function.apply(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SpectralAdaptiveScan(nn.Module):

    def __init__(
            self,
            channels,  
            kernel_size=3,  
            pad=1,
            group=1,
            offset_scale=0.1,
            act_layer='GELU',
            norm_layer='LN',
            spectral_reduction=False, 
            reduction_ratio=2, 
    ):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.group = group
        self.offset_scale = offset_scale
        self.spectral_reduction = spectral_reduction
        self.reduction_ratio = reduction_ratio

        if spectral_reduction:
            self.reduced_channels = channels // reduction_ratio
        else:
            self.reduced_channels = channels

        self.spectral_conv = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                stride=1,
                padding=pad,
                groups=channels 
            ),
            self._build_norm_layer(channels, norm_layer),
            self._build_act_layer(act_layer)
        )

        self.offset_generator = nn.Linear(
            channels,
            group * kernel_size 
        )

        self.weight_generator = nn.Linear(
            channels,
            group * kernel_size
        )

        if spectral_reduction:
            self.spectral_reduction_layer = nn.Conv1d(
                channels,
                self.reduced_channels,
                kernel_size=1
            )
            self.spectral_expansion_layer = nn.Conv1d(
                self.reduced_channels,
                channels,
                kernel_size=1
            )

        self._reset_parameters()

    def _build_norm_layer(self, dim, norm_layer):
        if norm_layer == 'BN':
            return nn.BatchNorm1d(dim)
        elif norm_layer == 'LN':
            return nn.BatchNorm1d(dim)
        else:
            return nn.Identity()

    def _build_act_layer(self, act_layer):
        if act_layer == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_layer == 'SiLU':
            return nn.SiLU(inplace=True)
        elif act_layer == 'GELU':
            return nn.GELU()
        else:
            return nn.Identity()

    def _reset_parameters(self):
        from torch.nn.init import constant_
        constant_(self.offset_generator.weight.data, 0.)
        constant_(self.offset_generator.bias.data, 0.)
        constant_(self.weight_generator.weight.data, 0.)
        constant_(self.weight_generator.bias.data, 0.)

    def spectral_adaptive_sample(self, x, offsets, weights):
        if len(x.shape) == 4:  
            B, C, H, W = x.shape
            x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
            offsets = offsets.view(B, H * W, -1)
            weights = weights.view(B, H * W, -1)

            sampled = self._adaptive_spectral_sample_1d(x_reshaped, offsets, weights)

            sampled = sampled.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        else: 
            sampled = self._adaptive_spectral_sample_1d(
                x.permute(0, 2, 1), offsets, weights
            ).permute(0, 2, 1)

        return sampled

    def _adaptive_spectral_sample_1d(self, x, offsets, weights):
        B, L, C = x.shape
        G = self.group
        K = self.kernel_size

        offsets = offsets.view(B, L, G, K)
        weights = weights.view(B, L, G, K)
        weights = F.softmax(weights, dim=-1) 

        group_size = C // G
        outputs = []

        for g in range(G):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size if g < G - 1 else C
            x_group = x[:, :, start_idx:end_idx] 

            offset_group = offsets[:, :, g, :]  
            weight_group = weights[:, :, g, :] 

            sampled_group = self._sample_spectral_neighbors(
                x_group, offset_group, weight_group
            )
            outputs.append(sampled_group)

        return torch.cat(outputs, dim=-1)

    def _sample_spectral_neighbors(self, x, offsets, weights):
        B, L, group_size = x.shape
        K = self.kernel_size

        base_indices = torch.arange(group_size, device=x.device, dtype=torch.float32)
        base_indices = base_indices.view(1, 1, 1, -1).expand(B, L, K, -1)

        offsets_expanded = offsets.unsqueeze(-1) 
        sample_indices = base_indices + offsets_expanded 

        sample_indices = torch.clamp(sample_indices, 0, group_size - 1)
        sampled_features = []
        for k in range(K):
            indices = sample_indices[:, :, k, :] 

            indices_floor = torch.floor(indices).long()
            indices_ceil = torch.ceil(indices).long()
            indices_frac = indices - indices_floor.float()

            indices_floor = torch.clamp(indices_floor, 0, group_size - 1)
            indices_ceil = torch.clamp(indices_ceil, 0, group_size - 1)

            x_floor = torch.gather(x, 2, indices_floor)
            x_ceil = torch.gather(x, 2, indices_ceil)
            x_interp = x_floor * (1 - indices_frac) + x_ceil * indices_frac

            sampled_features.append(x_interp)

        sampled_features = torch.stack(sampled_features, dim=2)
        weights_expanded = weights.unsqueeze(-1) 

        output = torch.sum(sampled_features * weights_expanded, dim=2)  

        return output

    def forward(self, input_feat, x):
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x_for_conv = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C).unsqueeze(-1)
            x_for_conv = x_for_conv.permute(0, 2, 1)

            spectral_feat = self.spectral_conv(x_for_conv)
            spectral_feat = spectral_feat.squeeze(-1)

            offsets = self.offset_generator(spectral_feat)
            weights = self.weight_generator(spectral_feat)

            offsets = offsets.view(B, H, W, -1)
            weights = weights.view(B, H, W, -1)

        else:
            B, C, L = x.shape
            spectral_feat = self.spectral_conv(x)
            spectral_feat = spectral_feat.permute(0, 2, 1)

            offsets = self.offset_generator(spectral_feat)
            weights = self.weight_generator(spectral_feat)

        output = self.spectral_adaptive_sample(x, offsets, weights)
        if self.spectral_reduction:
            if len(output.shape) == 4:
                B, C, H, W = output.shape
                output_flat = output.permute(0, 2, 3, 1).contiguous().view(-1, C).unsqueeze(-1)
                output_flat = output_flat.permute(0, 2, 1)

                reduced = self.spectral_reduction_layer(output_flat)
                expanded = self.spectral_expansion_layer(reduced)

                output = expanded.squeeze(-1).view(B, H, W, C).permute(0, 3, 1, 2)
            else:
                reduced = self.spectral_reduction_layer(output)
                output = self.spectral_expansion_layer(reduced)

        return output


class MS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.DWconv1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=3,
            padding=(3 - 1) // 2,
            **factory_kwargs,
        )

        self.DWconv2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=5,
            padding=(5 - 1) // 2,
            stride=2,
            **factory_kwargs,
        )

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=5,
            stride=2,
            padding=(5 - 1) // 2,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=False)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=False)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.da_scan = SpatialAdaptiveScan(channels=self.d_inner, group=1)
        random_seed_setting(6)


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):

        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=False):

        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D
    def zigzag_scan(self, x):
        B, C, H, W = x.shape
        device = x.device

        indices = []

        k = 0
        i = 0
        j = 0

        while i < H and j < W and k < H * W:
            indices.append(i * W + j)
            k += 1

            if (i + j) % 2 == 0:
                if (i - 1) >= 0 and (j + 1) >= W:
                    i = i + 1
                elif (i - 1) < 0 and (j + 1) < W:
                    j = j + 1
                elif (i - 1) < 0 and (j + 1) >= W:
                    i = i + 1
                else:
                    i = i - 1
                    j = j + 1
            elif (i + j) % 2 == 1:
                if (i + 1) < H and (j - 1) < 0:
                    i = i + 1
                elif (i + 1) >= H and (j - 1) >= 0:
                    j = j + 1
                elif (i + 1) >= H and (j - 1) < 0:
                    j = j + 1
                else:
                    i = i + 1
                    j = j - 1

        indices_tensor = torch.tensor(indices, device=device, dtype=torch.long)

        x_flat = x.view(B, C, H * W)
        x_zigzag = torch.gather(x_flat, dim=2, index=indices_tensor.unsqueeze(0).unsqueeze(0).expand(B, C, -1))

        return x_zigzag

    def forward_core(self, x: torch.Tensor):

        y = self.zigzag_scan(x)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        input_ = x.permute(0, 3, 1, 2).contiguous()

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous() 
        x = self.act(self.conv2d(x))

        x = self.da_scan(input_, x.permute(0, 2, 3, 1).contiguous())

        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class Fuse_SS2D(nn.Module):
    def __init__(
            self,
            d_model1,
            d_model2,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model1 = d_model1
        self.d_model2 = d_model2
        self.d_state = d_state
        self.expand = expand
        self.d_inner1 = int(self.expand * self.d_model1)
        self.d_inner2 = int(self.expand * self.d_model2)

        self.dt_rank = math.ceil(self.d_model1 / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner1, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner2, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) 
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner2, copies=4, merge=True) 
        self.Ds = self.D_init(self.d_inner2, copies=4, merge=True)  
        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner2)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D) 
        D._no_weight_decay = True
        return D

    def forward_core(self, x, y):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) 

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) 
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        y_hwwh = torch.stack([y.view(B, -1, L), torch.transpose(y, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        ys = torch.cat([y_hwwh, torch.flip(y_hwwh, dims=[-1])], dim=1) 
        ys = ys.float().view(B, -1, L)

        out_y = self.selective_scan(
            ys, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x, y):
        B, C, H, W = x.shape

        ya1, ya2, ya3, ya4 = self.forward_core(x, y)

        ya = ya1 + ya2 + ya3 + ya4
        ya = torch.transpose(ya, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        ya = self.out_norm(ya)
        return ya


class FSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim1: int = 0,
            hidden_dim2: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 0.5,
            d_conv=3,
            bias=False,
            conv_bias=True,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim1)
        self.ln_2 = norm_layer(hidden_dim2)
        self.expand = expand
        self.d_inner1 = int(self.expand * hidden_dim1)
        self.d_inner2 = int(self.expand * hidden_dim2)

        self.in_proj1 = nn.Linear(hidden_dim1, self.d_inner1 * 2, bias=bias)
        self.in_proj2 = nn.Linear(hidden_dim2, self.d_inner2 * 2, bias=bias)

        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner1,
            out_channels=self.d_inner1,
            groups=self.d_inner1,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.conv2d2 = nn.Conv2d(
            in_channels=self.d_inner2,
            out_channels=self.d_inner2,
            groups=self.d_inner2,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        self.attention1 = Fuse_SS2D(d_model1=hidden_dim1, d_model2=hidden_dim2, d_state=d_state, expand=expand,
                                    dropout=attn_drop_rate, **kwargs)
        self.attention2 = Fuse_SS2D(d_model1=hidden_dim2, d_model2=hidden_dim1, d_state=d_state, expand=expand,
                                    dropout=attn_drop_rate, **kwargs)
        self.out_proj1 = nn.Linear(self.d_inner1, hidden_dim1, bias=bias)
        self.out_proj2 = nn.Linear(self.d_inner2, hidden_dim2, bias=bias)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, y):
        dim_num = len(x.size())
        B, C, N, H, W = 0, 0, 0, 0, 0
        if (dim_num == 5):
            B, C, N, H, W = x.size()
            x = x.reshape(B, C * N, H, W)
        else:
            B, C, H, W = x.size()

        x = x.reshape(B, H, W, -1)
        y = y.reshape(B, H, W, -1)

        x_ = self.ln_1(x)
        y_ = self.ln_2(y)

        x_12 = self.in_proj1(x_)
        x_1, x_2 = x_12.chunk(2, dim=-1)

        y_12 = self.in_proj2(y_)
        y_1, y_2 = y_12.chunk(2, dim=-1)

        x_1 = x_1.reshape(B, -1, H, W)
        y_1 = y_1.reshape(B, -1, H, W)

        x_1 = self.act(self.conv2d1(x_1))
        y_1 = self.act(self.conv2d2(y_1))

        y_out = self.attention1(x_1, y_1)
        x_out = self.attention2(y_1, x_1)

        x_out = x_out * F.silu(x_2)
        y_out = y_out * F.silu(y_2)

        out_x = self.out_proj1(x_out)
        out_y = self.out_proj2(y_out)

        x = x + out_x
        y = y + out_y

        x = x.reshape(B, -1, H, W)
        y = y.reshape(B, -1, H, W)
        if (dim_num == 5):
            x = x.reshape(B, C, N, H, W)
        return x, y


class MSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 0.5,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = MS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input):
        cate = 'x'
        B, C, N, H, W = 0, 0, 0, 0, 0

        if (len(input.size()) == 5):
            B, C, N, H, W = input.size()
            input = input.reshape(B, C * N, H, W)
            cate = 'h'
        else:
            B, C, H, W = input.size()
        input = input.reshape(B, H, W, -1)
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = input + x
        x = x.permute(0, 3, 1, 2).contiguous()
        if (cate == 'h'):
            x = x.reshape(B, C, N, H, W)
        return x


class Spec_SS1D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.proj = nn.Linear(self.d_inner, self.d_inner, bias=bias)
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.da_scan = SpatialAdaptiveScan(channels=self.d_inner, group=1)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D) 
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, D = x.shape
        L = C
        K = 2
        xs = torch.stack([x.view(B, -1, L), torch.flip(x.view(B, -1, L), dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        y2 = torch.flip(out_y[:, 1], dims=[-1])
        return out_y[:, 0], y2

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = self.act(self.proj(x))

        y1, y2 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2
        y = torch.transpose(y, dim0=1, dim1=2).contiguous()
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SpecMambaBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 0.5,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = Spec_SS1D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate,
                                        **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input):
        B, C, N, H, W = input.size()
        input = input.reshape(B, C * N, H * W)

        x = self.ln_1(input)
        x = self.self_attention(x)
        x = x + input
        x = x.reshape(B, C, N, H, W)
        return x


class SpectralAdaptive_SS1D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=1.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            spectral_adaptive=True, 
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.spectral_adaptive = spectral_adaptive

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.proj = nn.Linear(self.d_inner, self.d_inner, bias=bias)
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        if self.spectral_adaptive:
            self.spectral_scan = SpectralAdaptiveScan(
                channels=self.d_inner,
                group=4,  
                **kwargs
            )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, D = x.shape
        L = C
        K = 2
        xs = torch.stack([x.view(B, -1, L), torch.flip(x.view(B, -1, L), dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float
        y2 = torch.flip(out_y[:, 1], dims=[-1])
        return out_y[:, 0], y2

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_inner = self.act(self.proj(x_inner))

        if self.spectral_adaptive:
            if len(x_inner.shape) == 3:  # [B, L, D]
                x_inner = x_inner.permute(0, 2, 1)  # [B, D, L]

            x_inner = self.spectral_scan(x_inner, x_inner)

            if len(x_inner.shape) == 3:  # [B, D, L]
                x_inner = x_inner.permute(0, 2, 1)  # [B, L, D]

        y1, y2 = self.forward_core(x_inner)
        assert y1.dtype == torch.float32
        y = y1 + y2
        y = torch.transpose(y, dim0=1, dim1=2).contiguous()
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SimpleSpecMambaBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer=nn.LayerNorm,
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 0.5,
            spectral_adaptive: bool = True,
            dropout: float = 0., 
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['dropout', 'attn_drop_rate']}

        self.self_attention = SpectralAdaptive_SS1D(
            d_model=hidden_dim,
            d_state=d_state,
            expand=expand,
            dropout=dropout,
            spectral_adaptive=spectral_adaptive,
            **clean_kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input):
        original_shape = input.shape

        if len(input.shape) == 5: 
            B, C, N, H, W = input.size()
            input = input.reshape(B, C * N, H * W)
            input = input.permute(0, 2, 1)  
            target_shape = (B, C, N, H, W)
        elif len(input.shape) == 4: 
            B, C, H, W = input.shape
            input = input.reshape(B, C, H * W)
            input = input.permute(0, 2, 1)  
            target_shape = (B, C, H, W)
        else:
            target_shape = original_shape

        x = self.ln_1(input)
        x = self.self_attention(x)
        x = input + self.drop_path(x)
        if len(original_shape) == 5:
            B, C, N, H, W = target_shape
            x = x.permute(0, 2, 1).reshape(B, C * N, H * W).reshape(B, C, N, H, W)
        elif len(original_shape) == 4:
            B, C, H, W = target_shape
            x = x.permute(0, 2, 1).reshape(B, C, H * W).reshape(B, C, H, W)
        return x


