# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# from gaussian import get_gaussian_kernel
#
# def SurDirect1(x, scale=4):
#     """
#     前向算子: HR -> LR
#     x: [B,C,H,W] 或 [H,W]
#     """
#     squeeze = False
#     if x.dim() == 2:
#         x = x.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
#         squeeze = True
#     elif x.dim() == 3:
#         x = x.unsqueeze(1)  # [B,1,H,W]
#         squeeze = True
#
#     B, C, H, W = x.shape
#     # 3x3 均值卷积
#     # kernel = torch.ones(1, 1, 3, 3) / 9.0
#     kernel = get_gaussian_kernel(kernel_size=3, sigma=1.0)
#     reflection_pad = nn.ReflectionPad2d(1)  # 3x3核需要1圈填充
#     x_padded = reflection_pad(x)
#     x_blur = F.conv2d(x_padded, kernel, padding=0, groups=C)
#     # 下采样
#     y = x_blur[:, :, ::scale, ::scale]
#
#     if squeeze:
#         y = y.squeeze(1)
#     return y
#
# def SurTranspose1(y, scale=4, mode='bicubic'):
#     squeeze = False
#     if y.dim() == 2:
#         y = y.unsqueeze(0).unsqueeze(0)
#         squeeze = True
#     elif y.dim() == 3:
#         y = y.unsqueeze(1)
#         squeeze = True
#
#     B, C, H, W = y.shape
#     y_up = F.interpolate(
#         y,
#         scale_factor=scale,
#         mode=mode,  # 双三次插值
#         align_corners=False  # 保证角落对齐
#     )
#     # 卷积转置保持反射填充（不修改，确保卷积边缘计算正确）
#     kernel = get_gaussian_kernel(kernel_size=3, sigma=1.0)
#     # kernel = torch.ones(1, 1, 3, 3) / 9.0
#     kernel = torch.flip(kernel, [2,3])
#     reflection_pad = nn.ReflectionPad2d(1)
#     y_up_padded = reflection_pad(y_up)
#     x_hat = F.conv2d(y_up_padded, kernel, padding=0, groups=C)
#
#     if squeeze:
#         x_hat = x_hat.squeeze(1)
#     return x_hat

import re
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2

# ---------- 10 个固定核（与你离线池一致） ----------
# 注意：这里仅保留需要的 10 个核。KID=k00..k09
FIXED_KERNEL_BANK = [
    ('gauss_iso',   {'size':5, 'sigma':0.8}),
    ('gauss_iso',   {'size':5, 'sigma':1.2}),
    ('gauss_aniso', {'size':5, 'sx':1.2, 'sy':0.8, 'angle':0.0}),
    ('gauss_aniso', {'size':5, 'sx':1.2, 'sy':0.8, 'angle':45.0}),
    ('motion',      {'size':5, 'length':3, 'angle':0.0}),
    ('motion',      {'size':5, 'length':3, 'angle':45.0}),
    ('disk',        {'size':5, 'radius':2.0}),
    ('sinc',        {'size':5, 'cutoff':0.25}),
    ('box',         {'size':5}),
    ('moffat',      {'size':5, 'alpha':1.5, 'beta':2.0}),
]

def k_gauss_iso(k, sigma):
    a = cv2.getGaussianKernel(k, sigma)
    ker = (a @ a.T).astype(np.float32)
    ker /= ker.sum(); return ker

def k_gauss_aniso(k, sx, sy, theta_deg):
    c = (k - 1) / 2.0
    y, x = np.mgrid[0:k, 0:k].astype(np.float32)
    x -= c; y -= c
    th = math.radians(theta_deg)
    xr =  x*math.cos(th) + y*math.sin(th)
    yr = -x*math.sin(th) + y*math.cos(th)
    ker = np.exp(-0.5*((xr/sx)**2 + (yr/sy)**2)).astype(np.float32)
    ker /= ker.sum(); return ker

def k_motion(k, length, angle_deg):
    ker = np.zeros((k,k), np.float32)
    c = (k - 1) / 2.0; ang = math.radians(angle_deg)
    for t in np.linspace(-length/2, length/2, num=length):
        x = int(round(c + t*math.cos(ang)))
        y = int(round(c + t*math.sin(ang)))
        if 0 <= x < k and 0 <= y < k: ker[y, x] = 1.0
    s = ker.sum(); ker = ker/s if s>0 else ker; return ker

def k_disk(k, radius):
    cy = cx = (k-1)/2.0
    y, x = np.mgrid[0:k, 0:k].astype(np.float32)
    mask = ((x-cx)**2 + (y-cy)**2) <= (radius**2)
    ker = mask.astype(np.float32)
    s = ker.sum(); ker = ker / s if s>0 else ker
    return ker

def k_sinc(k, cutoff=0.25):
    c = (k - 1) / 2.0
    y, x = np.mgrid[0:k, 0:k].astype(np.float32)
    x -= c; y -= c
    r = np.sqrt(x*x + y*y) + 1e-8
    s = np.sin(math.pi * cutoff * r) / (math.pi * cutoff * r)
    s[int(c), int(c)] = 1.0
    s = s.astype(np.float32); s /= s.sum(); return s

def k_box(k):
    ker = np.ones((k,k), np.float32)
    ker /= ker.sum(); return ker

def k_moffat(k, alpha=1.8, beta=2.2):
    c = (k-1)/2.0
    y, x = np.mgrid[0:k, 0:k].astype(np.float32)
    r = np.sqrt((x-c)**2 + (y-c)**2)
    ker = (1.0 + (r/alpha)**2) ** (-beta)
    ker = ker.astype(np.float32)
    ker /= ker.sum(); return ker

# OpenCV 的 1D 高斯（避免硬依赖 cv2）
def cv2_getGaussianKernel(ksize, sigma):
    # 与 OpenCV 行为一致的近似实现
    half = (ksize - 1) * 0.5
    x = np.arange(ksize, dtype=np.float64) - half
    if sigma <= 0:
        # OpenCV 的 sigma=0 时有特殊估计；这里给出安全值
        sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    g = np.exp(-(x**2)/(2*sigma*sigma))
    g /= g.sum()
    return g.reshape(-1,1)

def build_kernel_by_kid(kid: int):
    k_kind, cfg = FIXED_KERNEL_BANK[kid]
    k = cfg['size']
    if k_kind == 'gauss_iso':   ker = k_gauss_iso(k, cfg['sigma'])
    elif k_kind == 'gauss_aniso': ker = k_gauss_aniso(k, cfg['sx'], cfg['sy'], cfg['angle'])
    elif k_kind == 'motion':    ker = k_motion(k, cfg['length'], cfg['angle'])
    elif k_kind == 'disk':      ker = k_disk(k, cfg['radius'])
    elif k_kind == 'sinc':      ker = k_sinc(k, cfg['cutoff'])
    elif k_kind == 'box':       ker = k_box(k)
    elif k_kind == 'moffat':    ker = k_moffat(k, cfg['alpha'], cfg['beta'])
    else: raise ValueError(k_kind)
    ker_t = torch.from_numpy(ker).float().unsqueeze(0).unsqueeze(0)  # [1,1,k,k]
    return ker_t

def build_kernel_from_name(fname: str):
    """
    从 LR 文件名解析 kernel：
    1) 优先解析 KID=k00..k09
    2) 兼容旧命名：K=gauss_iso__SZ=..__SG=.. 等（可选）
    """
    # 1) KID 解析
    m = re.search(r'KID=k(\d{2})', fname)
    if m:
        kid = int(m.group(1))
        return build_kernel_by_kid(kid)

    # 2) 兼容旧命名（如有）
    m = re.search(r'K=([a-z_]+)__SZ=(\d+)(?:__SG=([\d.]+))?', fname)
    if m:
        kind = m.group(1); size = int(m.group(2))
        if kind == 'gauss_iso':
            sigma = float(m.group(3) or 1.2)
            ker = k_gauss_iso(size, sigma)
        elif kind == 'box':
            ker = k_box(size)
        else:
            # 只做最常用的兼容；其余 fallback 到一个温和高斯
            ker = k_gauss_iso(size, 1.2)
        ker_t = torch.from_numpy(ker).float().unsqueeze(0).unsqueeze(0)
        return ker_t

    # fallback：温和高斯
    ker = k_gauss_iso(15, 1.2)
    ker_t = torch.from_numpy(ker).float().unsqueeze(0).unsqueeze(0)
    return ker_t


# ========== 前向/后向算子（根据文件名核） ==========
def SurDirect1(x, scale=4, lr_name: str = None):
    """
    A: HR -> LR
    x: [B,C,H,W] / [C,H,W] / [H,W]
    lr_name: 用于解析卷积核（从 LR 文件名上取）
    """
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)    # [1,1,H,W]
        squeeze = True
    elif x.dim() == 3:
        x = x.unsqueeze(1)                 # [B,1,H,W]
        squeeze = True

    B, C, H, W = x.shape
    ker = build_kernel_from_name(lr_name or '')  # [1,1,k,k]
    pad = (ker.shape[-1] - 1) // 2
    ker = ker.to(x.device, x.dtype)

    # 反射填充 + 卷积
    x_blur = F.conv2d(F.pad(x, (pad,pad,pad,pad), mode='reflect'),
                      ker.repeat(C,1,1,1), padding=0, groups=C)
    # 步进下采样（与你的数据一致）
    y = x_blur[:, :, ::scale, ::scale]
    if squeeze:
        y = y.squeeze(1)
    return y


def SurTranspose1(y, scale=4, lr_name: str = None, mode='bicubic'):
    """
    lr_name: 用于解析相同的卷积核，然后用翻转核做反投影
    """
    squeeze = False
    if y.dim() == 2:
        y = y.unsqueeze(0).unsqueeze(0)
        squeeze = True
    elif y.dim() == 3:
        y = y.unsqueeze(1)
        squeeze = True

    B, C, h, w = y.shape
    # 上采样
    y_up = F.interpolate(y, scale_factor=scale, mode=mode, align_corners=False)

    # 反卷积（用翻转核）
    ker = build_kernel_from_name(lr_name or '')
    ker = torch.flip(ker, dims=[-1, -2])  # 空间翻转
    pad = (ker.shape[-1] - 1) // 2
    ker = ker.to(y_up.device, y_up.dtype)

    x_hat = F.conv2d(F.pad(y_up, (pad,pad,pad,pad), mode='reflect'),
                     ker.repeat(C,1,1,1), padding=0, groups=C)
    if squeeze:
        x_hat = x_hat.squeeze(1)
    return x_hat



