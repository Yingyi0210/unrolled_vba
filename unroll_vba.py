import torch.nn.init as init
import torch.nn as nn
import torch
import torch.nn.functional as F
from operators import *
import numpy as np
from torchvision.models import resnet18
import segmentation_models_pytorch as smp

class UnrollVBA(nn.Module):
    def __init__(self, block_num, scale, **kwargs):
        '''
        Unroll VBA
        '''
        super(UnrollVBA, self).__init__()
        self.height = kwargs.get('height', 256)
        self.width = kwargs.get('width', 256)
        self.scale = scale
        shared_r = rModule(scale)
        shared_unet = UNet(in_channels=3, out_channels=1)
        shared_resrefine = ResNetRefine(in_channels=3)
        self.model = nn.ModuleList([
            IterBlock(self.scale, r_block=shared_r,
                      unet_block=shared_unet,
                      resref_block=shared_resrefine)
            for _ in range(block_num)
        ])
        # self.model = nn.ModuleList([IterBlock(self.scale) for _ in range(block_num)])
        for idx, module in enumerate(self.model):
            # init.constant_(module.block1.gamma_n, 1e-5)
            # init.constant_(module.block1.gamma_p, 1e-5)
            init.uniform_(module.block1.gamma_n, a=0.0, b=1e-4)
            init.uniform_(module.block1.gamma_p, a=0.0, b=1e-4)
            # for m in module.block3.modules():
            #     if isinstance(m, nn.Conv2d):
            #         nn.init.kaiming_uniform_(m.weight, a=0.2)
            #         if m.bias is not None:
            #             nn.init.zeros_(m.bias)


        ### TODO : in train.py, we should prepare initialization, batch: [x_true, y, Aty]
        ### TODO : add loss in train.py

    def forward(self, Aty, diagAtA, mk_1, invSigmak_1, mk, invSigmak, lambdak):
        for module in self.model:
            mk_1, invSigmak_1, mk, invSigmak, lambdak = module(
                Aty, diagAtA, mk_1, invSigmak_1, mk, invSigmak, lambdak,
            )
        # print('mk ', mk.shape)
        return mk_1, invSigmak_1, mk, invSigmak, lambdak



"""### r"""

class rModule(nn.Module):
    '''
       Compute the forward of invSigma_r and m_r
       '''

    def __init__(self, scale):
        super(rModule, self).__init__()
        self.gamma_n = nn.Parameter(torch.Tensor(1))
        self.gamma_p = nn.Parameter(torch.Tensor(1))
        self.scale = scale
        # self.gamma_n = torch.nn.functional.softplus(nn.Parameter(torch.Tensor(1)))
        # self.gamma_p = torch.nn.functional.softplus(nn.Parameter(torch.Tensor(1)))

    def forward(self, diagAtA, Aty, mk, lambdak):
        diagDhtlambdakDh = lambdak + torch.roll(lambdak, -1, dims=3)
        diagDvtlambdakDv = lambdak + torch.roll(lambdak, -1, dims=2)
        diagDtlambdakD = diagDhtlambdakDh + diagDvtlambdakDv
        gamma_n = self.gamma_n
        gamma_p = self.gamma_p
        # gamma_n = F.softplus(self.gamma_n)
        # gamma_p = F.softplus(self.gamma_p)
        # print("gamma_n:", gamma_n.item(), "gamma_p:", gamma_p.item())
        # print('lambdak shape', lambdak.shape)
        # print("lambdak min/max:", lambdak.min().item(), lambdak.max().item())
        invSigma_r = gamma_n * diagAtA + gamma_p * diagDtlambdakD
        invSigma_r = F.relu(invSigma_r) + 1e-4
        # invSigma_r = torch.abs(self.gamma_n) * diagAtA + torch.abs(self.gamma_p) * diagDtlambdakD
        #invSigma_r = torch.abs(gamma_n) * diagAtA + torch.abs(gamma_p) * diagDtlambdakD
        # print(torch.max(diagAtA), torch.min(diagAtA))
        # def ensure_4d(mk):
        #     if mk.dim() == 3:  # [B,H,W]
        #         mk = mk.unsqueeze(1)
        #     elif mk.dim() == 4 and mk.shape[1] != 1:
        #         raise ValueError(f"Expected channel=1 but got {mk.shape}")
        #     return mk
        # mk = ensure_4d(mk)
        Amk = SurDirect1(mk, scale=self.scale)  # [B,1,H/scale,W/scale]
        # print("Amk shape:", Amk.shape)
        AtAmk = SurTranspose1(Amk, scale=self.scale)  # [B,1,H,W]
        # print("AtAmk shape:", AtAmk.shape)
        # operators
        Dhx = lambda x: x - torch.roll(x, 1, dims=3)
        Dvx = lambda x: x - torch.roll(x, 1, dims=2)
        Dhtx = lambda x: x - torch.roll(x, -1, dims=3)
        Dvtx = lambda x: x - torch.roll(x, -1, dims=2)

        # m_r = 1 / invSigma_r * (self.gamma_n * (Aty - AtAmk + diagAtA * mk)
        #                         - self.gamma_p * (Dhtx(lambdak * Dhx(mk)) + Dvtx(lambdak * Dvx(mk)))
        #                         + self.gamma_p * diagDtlambdakD * mk)
        m_r = 1 / invSigma_r * (gamma_n * (Aty - AtAmk + diagAtA * mk)
                                - gamma_p * (Dhtx(lambdak * Dhx(mk)) + Dvtx(lambdak * Dvx(mk)))
                                + gamma_p * diagDtlambdakD * mk)
        return m_r, invSigma_r


"""### lambda"""

class lambdaModule(nn.Module):
    '''
    Compute the forward of lambda
    '''

    def __init__(self):
        super(lambdaModule, self).__init__()

    def forward(self, mk, invSigmak):
        Dhx = lambda x: x - torch.roll(x, 1, dims=3)
        Dvx = lambda x: x - torch.roll(x, 1, dims=2)
        # print('invSigmak shape:', invSigmak.shape)
        traceDhtDhSigmak = 1 / invSigmak + torch.roll(1 / invSigmak, 1, dims=3)
        traceDvtDvSigmak = 1 / invSigmak + torch.roll(1 / invSigmak, 1, dims=2)
        lambdak = torch.pow(Dhx(mk), 2) + torch.pow(Dvx(mk), 2) + traceDhtDhSigmak + traceDvtDvSigmak
        return lambdak


# class DoubleConv(nn.Module):
#     """UNet 双卷积模块"""
#
#     def __init__(self, in_ch, out_ch):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             # nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
#
#
# class UNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=1):
#         super(UNet, self).__init__()
#
#         # 编码器（下采样部分）
#         self.inc = DoubleConv(in_channels, 64)
#         self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
#         self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
#         self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
#
#         # 解码器（上采样部分）
#         self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.conv_up1 = DoubleConv(512, 256)
#
#         self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.conv_up2 = DoubleConv(256, 128)
#
#         self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.conv_up3 = DoubleConv(128, 64)
#
#         # 输出层
#         self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
#         self.softplus = nn.Softplus()
#         self.relu = nn.ReLU()
#
#     def forward(self, mk1_invSig1, mk_invSig, mr_invSig_r):
#         # 编码路径
#         x = torch.cat([mk1_invSig1, mk_invSig, mr_invSig_r], dim=1)
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         # 解码路径
#         x = self.up1(x4)
#         x = torch.cat([x, x3], dim=1)
#         x = self.conv_up1(x)
#
#         x = self.up2(x)
#         x = torch.cat([x, x2], dim=1)
#         x = self.conv_up2(x)
#
#         x = self.up3(x)
#         x = torch.cat([x, x1], dim=1)
#         x = self.conv_up3(x)
#
#         out = self.outc(x)
#         out = self.softplus(out)
#         # out = self.relu(out)
#         return out

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        # 使用smp的UNet，指定编码器（可选轻量级如resnet18）
        self.model = smp.Unet(
            encoder_name="resnet18",  # 编码器 backbone，可选更轻量的如"resnet18"
            encoder_weights=None,  # 不使用预训练权重（任务无关）
            in_channels=in_channels,  # 输入3通道（与原逻辑一致）
            classes=out_channels,  # 输出1通道
            activation=None,  # 不使用默认激活，后续手动添加
        )
        self.softplus = nn.Softplus()  # 保留原有的非负激活

    def forward(self, mk1_invSig1, mk_invSig, mr_invSig_r):
        # 保持原输入拼接逻辑（3通道输入）
        x = torch.cat([mk1_invSig1, mk_invSig, mr_invSig_r], dim=1)
        # 调用smp的UNet前向传播
        out = self.model(x)
        # 应用Softplus确保输出非负（与原逻辑一致）
        out = self.softplus(out)
        return out

class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid, channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avgpool(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        return x * y

# 残差块
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, use_bn=False, use_ca=False, res_scale=0.2):
        super().__init__()
        pad = kernel_size // 2
        layers = [nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=True)]
        if use_bn:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=True))
        self.body = nn.Sequential(*layers)
        self.use_ca = use_ca
        self.ca = CALayer(channels) if use_ca else None
        self.res_scale = res_scale

    def forward(self, x):
        out = self.body(x)
        if self.use_ca:
            out = self.ca(out)
            return x + out
        else:
            return x + out * self.res_scale

# ResNetRefine：输入3通道 → 输出1通道
class ResNetRefine(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, num_blocks=8, use_ca=True, res_scale=0.2, use_bn=False):
        super().__init__()
        self.use_ca = use_ca
        # 输入层
        self.conv_in = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1, bias=True)
        # 残差体
        self.blocks = nn.ModuleList([
            ResBlock(base_ch, kernel_size=3, use_bn=use_bn, use_ca=use_ca, res_scale=res_scale)
            for _ in range(num_blocks)
        ])
        self.conv_out = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1, bias=True)
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.elu = nn.ELU(alpha=-1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        x: [B, 3, H, W]
        return: newinvSigmak > 0
        """
        x = self.conv_in(x)
        identity = x
        for blk in self.blocks:
            x = blk(x)
        x = x + identity  # 残差连接
        x = self.conv_out(x)
        # x = self.relu(x) + 1e-3
        x = self.elu(x) + 1e-6
        # x = self.softplus(x) + 1e-6
        return x

# class ResNetRefine(nn.Module):
#     def __init__(self, in_channels=3, base_ch=64, num_blocks=8, use_ca=True, res_scale=0.2, use_bn=False, alpha_max=0.2):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor(0.0))  # 学习步长的“logit”
#         self.alpha_max = alpha_max                   # 每步最大相对变化幅度
#         self.conv_in = nn.Conv2d(in_channels, base_ch, 3, padding=1)
#         self.blocks = nn.ModuleList([ResBlock(base_ch, use_bn=use_bn, use_ca=use_ca, res_scale=res_scale)
#                                      for _ in range(num_blocks)])
#         self.conv_out = nn.Conv2d(base_ch, 1, 3, padding=1)
#
#     def forward(self, x):  # x 仍是 concat(invSigmak, invSigmak_1, invSigma_r) or log域输入
#         h = self.conv_in(x)
#         for blk in self.blocks:
#             h = blk(h)
#         delta = self.conv_out(h)  # 无激活，R
#         # 把标量步长限制在 (0, alpha_max)
#         step = self.alpha_max * torch.sigmoid(self.alpha)
#         return delta, step


"""### iteration block"""

class IterBlock(nn.Module):
    # def __init__(self, scale):
    #     super(IterBlock, self).__init__()
    #     self.scale = scale
    #     self.block1 = rModule(scale)
    #     # self.block2 = UNetUpdateNet(in_channels=3, out_channels=1)
    #     self.block2 = UNet(in_channels=3, out_channels=1)
    #     self.block3 = ResNetRefine(in_channels=3)
    #     self.block4 = lambdaModule()
    def __init__(self, scale, r_block=None, unet_block=None, resref_block=None):
        super(IterBlock, self).__init__()
        self.scale = scale
        # 如果外部提供则复用，否则新建
        self.block1 = r_block if r_block is not None else rModule(scale)
        self.block2 = unet_block if unet_block is not None else UNet(in_channels=3, out_channels=1)
        self.block3 = resref_block if resref_block is not None else ResNetRefine(in_channels=3)
        self.block4 = lambdaModule()
        self.batch_counter = 0

    def forward(self, Aty, diagAtA, mk_1, invSigmak_1, mk, invSigmak, lambdak):
        # input:

        ## hyperparams: gamma_n, gamma_p: learned by network
        ## stepsize: s1, s2: learned by network

        ## constants: y, diagAtA, Dh, Dv
        ## image related: mk, invSigmak (mr, invSigmar), mk_1, invSigmak_1
        ## auxiliary variable: lambdak
        ## operators: A: projection, At: backpropagation??
        # output: new_mk, new_invSigmak
        self.batch_counter += 1
        m_r, invSigma_r = self.block1(diagAtA, Aty, mk, lambdak)
        # print("invSigma_r min/max:", invSigma_r.min().item(), invSigma_r.max().item())
        # _, _, step1, step2 = self.block2(mk_1, invSigmak_1, mk, invSigmak, m_r, invSigma_r)

        mk1_invSig1 = mk_1 * invSigmak_1
        mk_invSig = mk * invSigmak
        mr_invSig_r = m_r * invSigma_r
        newmk_mul_newinvSigmak = self.block2(mk1_invSig1, mk_invSig, mr_invSig_r)
        # print("newmk_mul_newinvSigmak min/max:", newmk_mul_newinvSigmak.min().item(), newmk_mul_newinvSigmak.max().item())
        net_input = torch.cat([invSigmak, invSigmak_1, invSigma_r], dim=1)
        # net_input = torch.cat([
        #         torch.log(invSigmak),
        #         torch.log(invSigmak_1),
        #         torch.log(invSigma_r)
        #     ], dim=1)
        # delta, step = self.block3(net_input)  # delta: [B,1,H,W]
        # print('delta min/max:', delta.min().item(), delta.max().item())
        # print('step min/max:', step.min().item(), step.max().item())
        # newinvSigmak = invSigmak * torch.exp(step * torch.tanh(delta))
        # newinvSigmak = newinvSigmak + 1e-6

        newinvSigmak = self.block3(net_input)
        newmk = newmk_mul_newinvSigmak/newinvSigmak
        # print("invSigmak min/max:", newinvSigmak.min().item(), newinvSigmak.max().item(), newinvSigmak.mean().item())
        # print("mk min/max:", newmk.min().item(), newmk.max().item(), newmk.mean().item())
        newlambdak = self.block4(newmk, newinvSigmak)
        mk_1 = mk
        invSigmak_1 = invSigmak
        mk = newmk
        invSigmak = newinvSigmak
        lambdak = newlambdak
        if self.batch_counter % 50 == 0:
            print(f"\n[Batch {self.batch_counter}] -------------------------------")
            print(f"mk       → min: {mk.min():.4e}, max: {mk.max():.4e}, mean: {mk.mean():.4e}")
            print(f"invSigmak→ min: {invSigmak.min():.4e}, max: {invSigmak.max():.4e}, mean: {invSigmak.mean():.4e}")
            print(f"invSigma_r→min: {invSigma_r.min():.4e}, max: {invSigma_r.max():.4e}, mean: {invSigma_r.mean():.4e}")
            print("newmk_mul_newinvSigmak min/max:", newmk_mul_newinvSigmak.min().item(),
                  newmk_mul_newinvSigmak.max().item())
            print("--------------------------------------------------------------")
        return mk_1, invSigmak_1, mk, invSigmak, lambdak







