import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from operators import SurTranspose1
from gaussian import get_gaussian_kernel
import torchvision.transforms.functional as TF
import torch.nn as nn
import cv2
import numpy as np

def get_high_detail_region(img_tensor, min_edge_count=50):
    """找到图像中边缘丰富的区域，返回可裁剪的top/left范围"""
    # 转换为numpy并边缘检测（注意：img_tensor是[0,1]范围，需转为[0,255]uint8）
    img_np = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)  # 关键：从[0,1]转为[0,255]
    edges = cv2.Canny(img_np, 50, 150)  # Canny边缘检测
    contour_list = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  # 提取轮廓

    # 统计每个区域的边缘密度（为边缘附近的像素加分）
    H, W = img_np.shape
    region_score = np.zeros((H, W), dtype=np.int32)
    for cnt in contour_list:
        for x, y in cnt[:, 0]:  # 遍历轮廓上的每个点
            if 0 < x < W - 1 and 0 < y < H - 1:  # 避免边界溢出
                # 为边缘点周围3x3区域加分（扩大边缘影响范围）
                region_score[y - 1:y + 2, x - 1:x + 2] += 1

    # 找到分数最高的区域，确定可裁剪范围
    max_score = region_score.max()
    if max_score < min_edge_count:  # 若最高分数低于阈值，说明无高细节区域，返回全图范围
        return 0, H, 0, W
    else:  # 否则返回分数≥50% max_score的区域范围
        y_coords, x_coords = np.where(region_score >= max_score * 0.5)
        min_y, max_y = y_coords.min(), y_coords.max()
        min_x, max_x = x_coords.min(), x_coords.max()
        return min_y, max_y, min_x, max_x


def save_image_pair(hr_img, lr_img, save_path, idx):
    """保存HR和LR图像对"""
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 转换为PIL图像（注意：需要将张量从[C, H, W]转换为[H, W, C]，并缩放至0-255）
    to_pil = transforms.ToPILImage()

    # 保存HR图像
    hr_pil = to_pil(hr_img)
    hr_pil.save(os.path.join(save_path, f"hr_{idx}.png"))

    # 保存LR图像
    lr_pil = to_pil(lr_img)
    lr_pil.save(os.path.join(save_path, f"lr_{idx}.png"))

# def random_photometric_augment(x):
#     """随机亮度/对比度/γ变换"""
#     brightness = random.uniform(0.8, 1.2)  # ±20%
#     contrast = random.uniform(0.8, 1.2)
#     gamma = random.uniform(0.8, 1.2)
#     x = torch.clamp(x * brightness, 0, 1)
#     mean = x.mean()
#     x = torch.clamp((x - mean) * contrast + mean, 0, 1)
#     x = torch.pow(x, gamma)
#     return x

def compute_adaptive_noise_std(img, alpha=0.1):
    """
    根据图像方差自适应确定噪声强度
    img: tensor, [0,1] 范围
    alpha: 比例系数 (0.05~0.2)
    """
    var = torch.var(img)
    noise_std = alpha * torch.sqrt(var)
    return noise_std.item()
# --- 前向退化函数 ---
def degradation_model_mean_kernel(x, scale=4, noise_std_range=(0.001, 0.05), sigma_range=(0.6, 2.0)):
    """
    严格符合 y = A x + n 的退化函数（内置 3x3 均值卷积核）
    Args:
        x (torch.Tensor): HR 图像，形状 [B, C, H, W]
        scale (int): 下采样因子
        noise_std (float): 高斯噪声标准差
    Returns:
        y (torch.Tensor): LR 图像，形状 [B, C, H/scale, W/scale]
    """
    B, C, H, W = x.shape

    # x = random_photometric_augment(x)

    # if random.random() < 0.3:
    #     angle = random.uniform(-3, 3)
    #     # x = TF.rotate(x, angle)
    #     x = TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.5)
    #
    # if random.random() < 0.3:
    #     tx = random.randint(-2, 2)
    #     ty = random.randint(-2, 2)
    #     x = TF.affine(x, angle=0, translate=(tx, ty), scale=1.0, shear=0)

    # Step 1: 3x3 均值卷积
    # kernel = torch.ones(1, 1, 3, 3)/ 9.0
    kernel = get_gaussian_kernel(kernel_size=3, sigma=1.0)
    reflection_pad = nn.ReflectionPad2d(1)  # 3x3核需要1圈填充
    x_padded = reflection_pad(x)
    x_blur = F.conv2d(x_padded, kernel, padding=0, groups=C)
    # Step 2: 下采样
    y_down = x_blur[:, :, ::scale, ::scale]
    # Step 3: 加噪声
    noise_std = compute_adaptive_noise_std(x, alpha=0.1)
    noise = torch.randn_like(y_down) * noise_std
    y_down = y_down + noise
    y_down = torch.clamp(y_down, 0.0, 1.0)
    return y_down



# --- Dataset 类 ---
class SRDataset(Dataset):
    def __init__(self, hr_folder, patch_size=256, scale=4, noise_std=0.01, augment=True, limit=None, save_samples=False, save_path="./samples"):
        self.hr_folder = hr_folder
        self.hr_files = [os.path.join(hr_folder, f) for f in os.listdir(hr_folder)
                         if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]
        if limit is not None and limit < len(self.hr_files):
            self.hr_files = self.hr_files[:limit]
        self.patch_size = patch_size
        self.scale = scale
        self.noise_std = noise_std
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        # 新增：保存样本相关参数
        self.save_samples = save_samples  # 是否保存样本
        self.save_path = save_path  # 保存路径
        self.saved_count = 0  # 已保存样本计数
        self.max_save = 20  # 最多保存5对样本（可修改）

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        # 读 HR 图像（保持灰度单通道）
        hr_img = Image.open(self.hr_files[idx]).convert("L")
        hr_tensor = self.to_tensor(hr_img)  # [C, H, W]

        C, H, W = hr_tensor.shape
        ps = (self.patch_size // self.scale) * self.scale
        if H < ps or W < ps:
            raise ValueError(f"图像太小，无法裁剪 {ps}×{ps} patch")

        # --- 随机裁剪 ---
        # top = random.randint(0, H - ps)
        # left = random.randint(0, W - ps)
        # hr_patch = hr_tensor[:, top:top + ps, left:left + ps].unsqueeze(0)  # [1,C,ps,ps]
        # ---高细节裁剪---
        # 1. 找到图像中的高细节区域范围
        min_y, max_y, min_x, max_x = get_high_detail_region(hr_tensor)
        # 2. 检查高细节区域是否足够大（能放下ps×ps的patch）
        valid_H = max_y - min_y
        valid_W = max_x - min_x
        if valid_H < ps or valid_W < ps:
            # 若高细节区域太小，退化为全图随机裁剪（避免报错）
            top = random.randint(0, H - ps)
            left = random.randint(0, W - ps)
        else:
            # 3. 在高细节区域内随机裁剪
            top = random.randint(min_y, max_y - ps)
            left = random.randint(min_x, max_x - ps)
        # 4. 裁剪并增加批量维度
        hr_patch = hr_tensor[:, top:top + ps, left:left + ps].unsqueeze(0)  # [1,C,ps,ps]
        # hr_patch = random_photometric_augment(hr_patch)
        # --- 数据增强 ---
        if self.augment:
            if random.random() < 0.5: hr_patch = torch.flip(hr_patch, [3])
            if random.random() < 0.5: hr_patch = torch.flip(hr_patch, [2])
            if random.random() < 0.5: hr_patch = hr_patch.transpose(2, 3)

        # --- 前向退化 ---
        lr_patch = degradation_model_mean_kernel(hr_patch, scale=self.scale)

        # --- 后向算子 ---
        Aty = SurTranspose1(lr_patch, scale=self.scale)

        if self.save_samples and self.saved_count < self.max_save:
            # 去除批量维度后保存
            save_image_pair(
                hr_img=hr_patch.squeeze(0),
                lr_img=lr_patch.squeeze(0),
                save_path=self.save_path,
                idx=self.saved_count
            )
            self.saved_count += 1
            print(f"已保存样本对 {self.saved_count}/{self.max_save} 到 {self.save_path}")

        return hr_patch.squeeze(0), lr_patch.squeeze(0), Aty.squeeze(0)



# 测试代码：生成并保存样本
if __name__ == "__main__":
    # 替换为你的HR图像文件夹路径
    hr_folder = r"D:\data\SR\DIV2K\DIV2K_train_HR_gray"  # 请修改为实际路径
    # 创建数据集实例，开启保存样本功能
    dataset = SRDataset(
        hr_folder=hr_folder,
        patch_size=256,
        scale=4,
        augment=True,
        save_samples=True,  # 开启保存
        save_path="./lr_hr_samples"  # 保存到这个文件夹
    )
    # 迭代几个样本触发保存（实际使用时会自动保存前5个）
    for i in range(min(20, len(dataset))):
        dataset[i]
