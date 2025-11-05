import torch

# def get_gaussian_kernel(kernel_size=3, sigma=1.0, channels=1):
#     """
#     构造高斯卷积核 (归一化后 sum=1)
#     """
#     # 坐标
#     x_coord = torch.arange(kernel_size)
#     x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
#     y_grid = x_grid.t()
#     xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
#
#     mean = (kernel_size - 1) / 2.
#     variance = sigma ** 2.
#
#     # 高斯公式
#     gaussian_kernel = (1. / (2. * torch.pi * variance)) * \
#                       torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
#
#     # 归一化
#     gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
#
#     # [out_channels, in_channels, kH, kW]
#     gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
#     gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
#     return gaussian_kernel
def get_gaussian_kernel(kernel_size=3, sigma=1.0, channels=1):
    """构造高斯卷积核 (归一化后 sum=1)"""
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * torch.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    return gaussian_kernel