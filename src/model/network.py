import os
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Union

# imports: 确保路径与项目一致
# 假设 Stage1 和 Stage2 的导入路径正确
from src.model.stage1_synthesis import SynthesisModule
from src.model.stage2_color_refined import Stage2ColorRefine

# 导入常用 prior 实现
from src.model.priors import *


class RefinedColorCorrectionNet(nn.Module):
    def __init__(self,
                 prior_modules=None,
                 device: torch.device = torch.device('cpu'),
                 in_ch: int = 3,
                 cose_kernel: int = 3,
                 como_dim: int = 64,
                 dm_in_ch: int = 320,
                 dm_downsample_factor: int = 8):
        super().__init__()

        # 构建 prior_modules 列表
        if prior_modules is None:
            prior_modules = [
                GrayWorldCompensator(),
                UDCPColorRestorationHighQuality(),
                # ... 其他传统 prior
            ]

        # Stage1
        self.stage1 = SynthesisModule(prior_modules)

        # Stage2
        num_priors = len(prior_modules)

        self.stage2 = Stage2ColorRefine(
            cose_kernel=cose_kernel,
            num_priors=num_priors
        )

        # 保存 DM 特征的维度信息，用于 forward 函数中检查或辅助调试
        self.dm_in_ch = dm_in_ch
        self.dm_downsample_factor = dm_downsample_factor

    def forward(self,
                x: torch.Tensor,
                ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            x: (B,3,H,W) 原始输入图像 Ix
            F_DM_l: (B, DM_in_ch, H/F, W/F) 预先计算好的 DM 骨干特征

        Returns:
            Dict of results, including I_refined (最终精修图像)
        """
        # ------------------------------------
        # 1. Stage1: Prior Synthesis
        # ------------------------------------
        # 输出：I_synthesis (语义先验), residual_list (几何先验), ...
        I_synthesis, attn_stage1, residual_list, prior_list = self.stage1(x)
        # residual_list: List[(B,3,H,W)]

        # ------------------------------------
        # 2. Stage2: Color Refinement
        # ------------------------------------
        # Stage2 forward 签名: Ix, residual_list, I_synthesis, F_DM_l
        I_refined, weighted_offset_list, como_out = self.stage2(
            Ix=x,
            prior_list=prior_list,
            I_synthesis=I_synthesis
            # 移除 F_DM_l 参数
        )

        # 组织返回（方便训练脚本解包）
        return {
            "I_refined": I_refined,
            "I_synthesis": I_synthesis,
            "attn_stage1": attn_stage1,
            "attn_stage2": None,  # Stage 2 当前未返回 attn
            "residuals_stage1": residual_list,
            "offsets_stage2": weighted_offset_list,
            "priors": prior_list,
            "como_out": como_out
        }