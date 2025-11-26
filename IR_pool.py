import os, math, json, argparse
from glob import glob
from functools import partial
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# ========== 内容优先采样：热力图 + 加权取样 ==========

def compute_saliency_map(hr01,
                         black_thr=0.05,
                         high_thr_mode='percentile',  # 'percentile' or 'absolute'
                         high_thr_value=99.5,         # 百分位数，比如99.5分位
                         abs_high_thr=0.98,           # 当 high_thr_mode='absolute' 时使用
                         var_ks=9, grad_ks=3,
                         w_grad=0.6, w_var=0.3, w_nonblack=0.1,
                         w_bright_suppress=0.25):     # 对高亮区域的抑制权重（0~1，越大压得越狠）
    """
    返回 saliency ∈ [0,1]，越大越有内容。会同时抑制纯黑与近饱和高亮区。
    """
    H, W = hr01.shape

    # 非黑掩码
    nonblack = (hr01 > black_thr).astype(np.float32)

    # 亮度上限阈值（自适应或绝对）
    if high_thr_mode == 'percentile':
        hi = float(np.percentile(hr01, high_thr_value))
    else:
        hi = float(abs_high_thr)
    too_bright = (hr01 >= hi).astype(np.float32)  # 近饱和区域

    # Sobel 梯度
    gx = cv2.Sobel(hr01, cv2.CV_32F, 1, 0, ksize=grad_ks)
    gy = cv2.Sobel(hr01, cv2.CV_32F, 0, 1, ksize=grad_ks)
    grad = np.sqrt(gx*gx + gy*gy)
    if grad.max() > 0: grad = grad / (grad.max() + 1e-8)

    # 局部方差
    mean  = cv2.boxFilter(hr01, -1, (var_ks, var_ks), normalize=True)
    mean2 = cv2.boxFilter(hr01*hr01, -1, (var_ks, var_ks), normalize=True)
    var = np.clip(mean2 - mean*mean, 0.0, None)
    if var.max() > 0: var = var / (var.max() + 1e-8)

    # 基础显著性
    sal = w_grad*grad + w_var*var + w_nonblack*nonblack

    # 对“过亮”区域做抑制（1 - w * mask）
    if w_bright_suppress > 0:
        sal *= (1.0 - w_bright_suppress * too_bright)

    sal = cv2.GaussianBlur(sal, (0,0), 1.0)
    m, M = float(sal.min()), float(sal.max())
    sal = (sal - m) / (M - m + 1e-8)
    return sal


def sample_patch_from_saliency(saliency, ps, rng, min_peak=0.1, gamma=2.0):
    """
    根据 saliency 计算窗口得分并按权重采样一个 (top,left)
    - min_peak: 全图最高窗口分数若 < min_peak，则回退随机
    - gamma: 提高对高分区的偏好（>1 更“贪心”）
    """
    H, W = saliency.shape
    if H < ps or W < ps:
        raise ValueError(f'image too small for {ps}x{ps}: {H}x{W}')

    # 窗口得分：对 saliency 做“盒滤”得到每个 ps×ps 窗口的总分
    win_sum = cv2.boxFilter(saliency, ddepth=-1, ksize=(ps, ps), normalize=False)
    # 只保留有效放置区域（顶点范围）
    win_valid = win_sum[:H - ps + 1, :W - ps + 1]
    vmax = float(win_valid.max())
    if vmax < min_peak:
        # 太“黑”，回退随机
        top  = int(rng.integers(0, H - ps + 1))
        left = int(rng.integers(0, W - ps + 1))
        return top, left

    # 按权重采样
    w = np.power(np.maximum(win_valid, 0.0), gamma)
    s = w.sum()
    if s <= 0:
        top  = int(rng.integers(0, H - ps + 1))
        left = int(rng.integers(0, W - ps + 1))
        return top, left
    probs = (w / s).ravel()
    idx = int(rng.choice(probs.size, p=probs))
    r = idx // w.shape[1]
    c = idx %  w.shape[1]
    return int(r), int(c)

def pick_patch_with_constraints(img01, sal, ps, rng,
                               mean_lo=0.03, mean_hi=0.95,
                               std_lo=0.02, max_trials=20):
    """
    在 saliency 加权采样得到 (top,left) 后，做窗口均值/方差约束；
    不满足则重采，最多 max_trials 次，最后一次回退随机。
    """
    H, W = img01.shape
    for t in range(max_trials):
        top, left = sample_patch_from_saliency(sal, ps, rng, min_peak=0.1, gamma=2.0)
        patch = img01[top:top+ps, left:left+ps]
        m = float(patch.mean())
        s = float(patch.std())
        if (mean_lo <= m <= mean_hi) and (s >= std_lo):
            return top, left
    # 回退：仍不满足就随机
    top  = int(rng.integers(0, H - ps + 1))
    left = int(rng.integers(0, W - ps + 1))
    return top, left

# ---------------- 基础 I/O ----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_gray_float(path):
    img = Image.open(path).convert('L')     # 红外：单通道
    arr = np.array(img).astype(np.float32) / 255.0
    return arr  # HxW, [0,1]

def save_img_gray_u8(path, x, webp_lossless=True):
    x = np.clip(x, 0.0, 1.0)
    u8 = (x * 255.0 + 0.5).astype(np.uint8)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.webp':
        if webp_lossless and hasattr(cv2, 'IMWRITE_WEBP_LOSSLESS'):
            cv2.imwrite(path, u8, [cv2.IMWRITE_WEBP_LOSSLESS, 1])  # 真·无损
        else:
            cv2.imwrite(path, u8, [cv2.IMWRITE_WEBP_QUALITY, 100])  # 高质量有损
    else:
        cv2.imwrite(path, u8)

def rand_patch_coords(H, W, ps, rng):
    if H < ps or W < ps:
        raise ValueError(f'image too small for {ps}x{ps}: {H}x{W}')
    top  = int(rng.integers(0, H - ps + 1))
    left = int(rng.integers(0, W - ps + 1))
    return top, left

def reflect_conv(x, ker):
    k = ker.shape[0]
    pad = (k - 1) // 2
    xpad = cv2.copyMakeBorder(x, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    y = cv2.filter2D(xpad, -1, ker, borderType=cv2.BORDER_CONSTANT)
    return y[pad:-pad, pad:-pad]

def downsample_stride(x, s):  # 严格整数下采样（与 A 的定义一致）
    return x[::s, ::s]

# ---------------- 需要的核构造（仅保留 10 个会用到的种类） ----------------
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
    s = s.astype(np.float32)
    s /= s.sum()
    return s

def k_box(k):
    ker = np.ones((k,k), np.float32)
    ker /= ker.sum();
    return ker

def k_moffat(k, alpha=1.8, beta=2.2):
    c = (k-1)/2.0
    y, x = np.mgrid[0:k, 0:k].astype(np.float32)
    r = np.sqrt((x-c)**2 + (y-c)**2)
    ker = (1.0 + (r/alpha)**2) ** (-beta)
    ker = ker.astype(np.float32)
    ker /= ker.sum()
    return ker

# ---------------- 10 个固定核：id -> (类型, 参数) ----------------
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



def build_kernel_by_kid(kid:int):
    k_kind, k_cfg = FIXED_KERNEL_BANK[kid]
    k = k_cfg['size']
    if k_kind == 'gauss_iso':   ker = k_gauss_iso(k, k_cfg['sigma'])
    elif k_kind == 'gauss_aniso': ker = k_gauss_aniso(k, k_cfg['sx'], k_cfg['sy'], k_cfg['angle'])
    elif k_kind == 'motion':    ker = k_motion(k, k_cfg['length'], k_cfg['angle'])
    elif k_kind == 'disk':      ker = k_disk(k, k_cfg['radius'])
    elif k_kind == 'sinc':      ker = k_sinc(k, k_cfg['cutoff'])
    elif k_kind == 'box':       ker = k_box(k)
    elif k_kind == 'moffat':    ker = k_moffat(k, k_cfg['alpha'], k_cfg['beta'])
    else: raise ValueError(k_kind)
    return ker, k_kind, k_cfg

def make_kid_str(kid:int):
    return f"k{kid:02d}"  # 文件名中的 KID 片段

# ---------------- 噪声（保留即可；不写入文件名，仅入 manifest） ----------------
def sample_noise_by_snr(rng, snr_list=[45 ,35, 25, 15], mode='random'):
    """
    mode: 'random' -> 随机从 snr_list 中采样；
          'balanced' -> 依顺序 / 固定策略（可用于分 SNR 生成 dataset）
    返回 {'type':'gauss', 'snr_db': val}
    """
    if mode == 'random':
        snr = float(rng.choice(snr_list))
    else:
        # 你可以改成按比例取值或按 idx 取值。这里默认 random 以简洁为主。
        snr = float(rng.choice(snr_list))
    return {'type': 'gauss', 'snr_db': snr}

# 添加噪声：根据 lr0 的能量和所给 snr 计算 sigma，然后添加零均值高斯噪声
def add_noise_with_snr(lr_image, noise_cfg, rng):
    """
    lr_image: float32 array in [0,1] (建议传入 blur 后并下采样的 lr0)
    noise_cfg: {'type':'gauss', 'snr_db': float}
    """
    if noise_cfg['type'] != 'gauss':
        raise ValueError("only 'gauss' supported in this function")

    # 计算信号功率（均方）：按图像本身（如果多通道需按通道均值）
    P_signal = float(np.mean(lr_image.astype(np.float32)**2) + 1e-12)
    snr_db = float(noise_cfg['snr_db'])
    sigma2 = P_signal / (10.0 ** (snr_db / 10.0))
    sigma = math.sqrt(sigma2)

    noise = rng.normal(0.0, sigma, lr_image.shape).astype(np.float32)
    y = lr_image + noise
    return np.clip(y, 0.0, 1.0), float(sigma)

# ---------------- 主处理 ----------------
def process_one_image(pack, hr_path):
    out_root, scale, ps, P, M, use_webp, seed_base = pack
    rng = np.random.default_rng(seed_base + (hash(os.path.basename(hr_path)) & 0x7fffffff))
    SNR_LIST = [45, 35, 25, 15]  # 固定四种 SNR
    hr = load_gray_float(hr_path)
    H, W = hr.shape
    if H < ps or W < ps:
        print(f"Skip image {hr_path}: too small ({H}x{W})")
        return []  # 直接跳过
    name = os.path.splitext(os.path.basename(hr_path))[0]
    sal = compute_saliency_map(hr, black_thr=0.05, var_ks=9, grad_ks=3,
                               w_grad=0.6, w_var=0.3, w_nonblack=0.1)

    dir_hrp = os.path.join(out_root, 'HR_patches', name)
    dir_lrp = os.path.join(out_root, 'LR', name)
    ensure_dir(dir_hrp); ensure_dir(dir_lrp)

    ext = '.webp' if use_webp else '.png'
    rows = []

    for pidx in range(P):
        top, left = pick_patch_with_constraints(hr, sal, ps, rng,
                                                mean_lo=0.03, mean_hi=0.95,
                                                std_lo=0.02, max_trials=20)
        hr_patch = hr[top:top+ps, left:left+ps]
        hrp_name = f'{name}_p{pidx:05d}_t{top}_l{left}{ext}'
        hrp_path = os.path.join(dir_hrp, hrp_name)
        save_img_gray_u8(hrp_path, hr_patch, webp_lossless=use_webp)

        for vidx in range(M):
            kid = vidx % 10                          # 10 个固定核
            ker, k_kind, k_cfg = build_kernel_by_kid(kid)
            blur = reflect_conv(hr_patch, ker)
            lr0  = downsample_stride(blur, scale)
            for snr in SNR_LIST:
                n_cfg = {'type': 'gauss', 'snr_db': snr}  # 固定 SNR
                lr, sigma = add_noise_with_snr(lr0, n_cfg, rng)

                kid_str = make_kid_str(kid)
                snr_str = f"SNR{int(snr)}"
                lr_name = f"{name}_p{pidx:05d}_v{vidx:02d}__KID={kid_str}__{snr_str}{ext}"
                lr_path = os.path.join(dir_lrp, lr_name)
                save_img_gray_u8(lr_path, lr, webp_lossless=use_webp)

                rows.append({
                    'hr_image': os.path.relpath(hr_path, out_root),
                    'hr_patch': os.path.relpath(hrp_path, out_root),
                    'lr_image': os.path.relpath(lr_path, out_root),
                    'scale': scale,
                    'patch': {'top': int(top), 'left': int(left), 'size': int(ps)},
                    'kid': kid,                     # 0..9
                    'kid_str': kid_str,             # "k00".."k09"
                    'kernel': {'type': k_kind, **{k: (float(v) if isinstance(v,(int,float)) else v)
                                                  for k,v in k_cfg.items()}},
                    'noise': {
                        'type': n_cfg['type'],   # 'gauss'
                        'snr_db': float(n_cfg['snr_db']),
                        'sigma': float(sigma),
                    },
                })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hr_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--scale', type=int, default=4)
    ap.add_argument('--patch_size', type=int, default=256)
    ap.add_argument('--patches_per_image', type=int, default=50)   # P
    ap.add_argument('--variants_per_patch', type=int, default=10)  # M=10 对应 10 个固定核
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--save_webp', action='store_true')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--limit', type=int, default=-1, help='只取前K张HR，-1为全部')
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    exts = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp')
    hr_paths = []
    for e in exts:
        hr_paths += glob(os.path.join(args.hr_dir, f'*{e}'))
    hr_paths = sorted(hr_paths)
    if args.limit and args.limit > 0:
        hr_paths = hr_paths[:args.limit]
    if len(hr_paths)==0:
        raise RuntimeError('HR 文件夹为空')

    pack = (args.out_dir, args.scale, args.patch_size,
            args.patches_per_image, args.variants_per_patch,
            args.save_webp, args.seed)

    all_rows = []
    with Pool(processes=args.workers) as pool:
        worker = partial(process_one_image, pack)
        # for rows in tqdm(pool.imap_unordered(worker, hr_paths), total=len(hr_paths)):
        for rows in tqdm(pool.imap_unordered(worker, hr_paths), total=len(hr_paths), dynamic_ncols=True,
                             mininterval=0.2):
            all_rows.extend(rows)

    mani = os.path.join(args.out_dir, 'manifest.jsonl')
    with open(mani, 'w', encoding='utf-8') as f:
        for r in all_rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')

    summary = {
        'hr_dir': os.path.abspath(args.hr_dir),
        'out_dir': os.path.abspath(args.out_dir),
        'scale': args.scale,
        'patch_size': args.patch_size,
        'patches_per_image': args.patches_per_image,
        'variants_per_patch': args.variants_per_patch,
        'num_hr_images': len(hr_paths),
        'num_hr_patches': len(hr_paths)*args.patches_per_image,
        'num_lr_images': len(hr_paths)*args.patches_per_image*args.variants_per_patch,
        'save_webp': bool(args.save_webp),
        'seed': args.seed
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('Done.\nSummary:', summary)

if __name__ == '__main__':
    main()


# python IR_pool.py --hr_dir "D:/data/SR/train/CoRPLE M3FD_infrared/Infrared" --out_dir "D:/data/SR/train_LR_HR_5x5" --scale 4 --patch_size 256 --patches_per_image 5 --variants_per_patch 10 --workers 8 --save_webp

