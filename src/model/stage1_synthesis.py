import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.TCTL.tctl_cyan_module import TCTLCyanModule
from src.model.physics.ScatteringModule import TNet, JNet, GlobalLightEstimator
from src.model.prior import *


# ----------------------------------------------
# Generic Attention Fusion
# ----------------------------------------------
class MultiAttention(nn.Module):
    """
    根据 prior 数量动态创建 N 个注意力通道
    """
    def __init__(self, num_priors, in_ch=3):
        super().__init__()
        self.num_priors = num_priors

        self.attn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
            )
            for _ in range(num_priors)
        ])

    def forward(self, x):
        attn_list = []

        for layer in self.attn_layers:
            attn_list.append(layer(x))  # 每个是 (B,1,H,W)

        A_sum = sum(attn_list) + 1e-6
        attn_list = [A / A_sum for A in attn_list]

        return attn_list  # List[(B,1,H,W)]


# ----------------------------------------------
# Stage1: SynthesisModule
# 支持添加任意数量的 prior 模块
# ----------------------------------------------
class SynthesisModule(nn.Module):
    def __init__(self, prior_modules):
        """
        prior_modules: List[nn.Module]
        每个 prior_module(x) -> enhanced_image (B,3,H,W)
        """
        assert len(prior_modules) > 0, "SynthesisModule requires >=1 prior module"

        super().__init__()

        self.prior_modules = nn.ModuleList(prior_modules)
        self.num_priors = len(prior_modules)

        self.attn = MultiAttention(num_priors=self.num_priors, in_ch=3)

    def forward(self, x):
        """
        输入:
            x: 原图 (B,3,H,W)

        输出:
            I_synthesis : (B,3,H,W)
            attn_list   : List[(B,1,H,W)]
            residual_list : List[(B,3,H,W)]
            prior_list  : List[(B,3,H,W)]
        """

        device = x.device  # 获取输入 x 的设备 (cuda:0)

        # ------------------------
        # 1. 计算所有 Prior 输出
        # ------------------------
        prior_list = [module(x) for module in self.prior_modules]
        # e.g. [I_gray, I_udcp, I_lab]

        # 强制将所有 Prior 输出移动到 GPU (快速修复)
        prior_list = [Pi.to(device) for Pi in prior_list]  # <--- 添加这一行

        # ------------------------
        # 2. N 通道注意力
        # ------------------------
        attn_list = self.attn(x)
        # 长度 = num_priors

        # ------------------------
        # 3. 合成 Synthesis
        # ------------------------
        # safer init: zeros tensor with same device/dtype as priors
        I_synthesis = torch.zeros_like(prior_list[0])
        for Ai, Pi in zip(attn_list, prior_list):
            I_synthesis = I_synthesis + Ai * Pi
        # optional clamp
        # I_synthesis = torch.clamp(I_synthesis, 0.0, 1.0)

        # ------------------------
        # 4. residuals (每个 prior 对应一个 residual)
        # ------------------------
        residual_list = [torch.clamp(Pi - x, -1, 1) for Pi in prior_list]

        return I_synthesis, attn_list, residual_list, prior_list