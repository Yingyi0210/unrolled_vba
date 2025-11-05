import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
from gaussian import get_gaussian_kernel

def SurDirect1(x, scale=4):
    """
    前向算子: HR -> LR
    x: [B,C,H,W] 或 [H,W]
    """
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        squeeze = True
    elif x.dim() == 3:
        x = x.unsqueeze(1)  # [B,1,H,W]
        squeeze = True

    B, C, H, W = x.shape
    # 3x3 均值卷积
    # kernel = torch.ones(1, 1, 3, 3) / 9.0
    kernel = get_gaussian_kernel(kernel_size=3, sigma=1.0)
    reflection_pad = nn.ReflectionPad2d(1)  # 3x3核需要1圈填充
    x_padded = reflection_pad(x)
    x_blur = F.conv2d(x_padded, kernel, padding=0, groups=C)
    # 下采样
    y = x_blur[:, :, ::scale, ::scale]

    if squeeze:
        y = y.squeeze(1)
    return y


# def SurTranspose1(y, scale=4, mode='bicubic'):
#     """
#     后向算子: LR -> HR (卷积转置)
#     y: [B,C,H,W] 或 [H,W]
#     """
#     squeeze = False
#     if y.dim() == 2:
#         y = y.unsqueeze(0).unsqueeze(0)
#         squeeze = True
#     elif y.dim() == 3:
#         y = y.unsqueeze(1)
#         squeeze = True
#
#     B, C, H, W = y.shape
#     # 上采样
#     # y_up = F.interpolate(y, scale_factor=scale, mode=mode, align_corners=True)
#     # y_up = y_up.clamp(0.0, 1.0)
#     y_up = torch.zeros(B, C, H*scale, W*scale, device=y.device, dtype=y.dtype)
#     y_up[:, :, ::scale, ::scale] = y
#
#     # 卷积转置（3x3 均值核翻转）
#     # kernel = torch.ones((1, 1, 3, 3), device=y.device, dtype=y.dtype) / 9.0
#     kernel = get_gaussian_kernel(kernel_size=3, sigma=1.0)
#     kernel = torch.flip(kernel, [2,3])
#     pad = 1
#     reflection_pad = nn.ReflectionPad2d(pad)
#     y_up_padded = reflection_pad(y_up)  # 填充后尺寸：(H*scale + 2*pad) × (W*scale + 2*pad)
#     x_hat = F.conv2d(y_up_padded, kernel, padding=0,
#                      groups=C)  # 卷积后尺寸：(H*scale + 2*pad - kernel_size + 1) → 即 H*scale（因为 kernel_size=3）
#     if squeeze:
#         x_hat = x_hat.squeeze(1)
#     return x_hat

def SurTranspose1(y, scale=4, mode='bicubic'):
    squeeze = False
    if y.dim() == 2:
        y = y.unsqueeze(0).unsqueeze(0)
        squeeze = True
    elif y.dim() == 3:
        y = y.unsqueeze(1)
        squeeze = True

    B, C, H, W = y.shape
    y_up = F.interpolate(
        y,
        scale_factor=scale,
        mode=mode,  # 双三次插值
        align_corners=False  # 保证角落对齐
    )
    # 卷积转置保持反射填充（不修改，确保卷积边缘计算正确）
    kernel = get_gaussian_kernel(kernel_size=3, sigma=1.0)
    # kernel = torch.ones(1, 1, 3, 3) / 9.0
    kernel = torch.flip(kernel, [2,3])
    reflection_pad = nn.ReflectionPad2d(1)
    y_up_padded = reflection_pad(y_up)
    x_hat = F.conv2d(y_up_padded, kernel, padding=0, groups=C)

    if squeeze:
        x_hat = x_hat.squeeze(1)
    return x_hat


