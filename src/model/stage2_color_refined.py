# src/model/stage2_color_refined.py
# ============================================================
# Stage2: Color Refine (PURE Pixel-level Fusion)
# - NO encoder
# - NO decoder
# - NO rejector
# - Direct RGB fusion via COMO-Mamba
# ============================================================

import torch
import torch.nn as nn
from typing import List

from src.model.dconv.cose import ColorShiftEstimation
from src.model.dconv.como_mamba import COMO_Mamba
from src.model.module.cas_offset import OffsetCAS


class Stage2ColorRefine(nn.Module):
    """
    Pure pixel-level Stage2 refinement.
    All fusion and refinement are performed directly in RGB space.
    """

    def __init__(
        self,
        cose_kernel: int = 3,
        num_priors: int = 4
    ):
        super().__init__()

        self.cose_kernel = cose_kernel
        self.cas = OffsetCAS(offset_ch=3)
        self.last_cas_weights: torch.Tensor | None = None
        self.num_priors = num_priors
        self.cose_list = nn.ModuleList([
            ColorShiftEstimation(
                in_channels=3,
                kernel_size=self.cose_kernel,
                modulation=True
            )
            for _ in range(num_priors)
        ])

        # --------------------------------------------------
        # COMO-Mamba
        # - in_ch = 3 (RGB)
        # - offset_ch = 3 (RGB offsets)
        # - dim = 64 (internal hidden only)
        # --------------------------------------------------
        self.como = COMO_Mamba(
            offset_ch=3,
            dim=64,
            n_offsets=num_priors + 1,
            use_multidirectional=True,
            md_directions=4,
        )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(
        self,
        Ix: torch.Tensor,
        prior_list: List[torch.Tensor],
        I_synthesis: torch.Tensor
    ):
        """
        Args:
            Ix : (B, 3, H, W) original input image
            prior_list : list of (B, 3, H, W) Stage1 priors
            I_synthesis : (B, 3, H, W) Stage1 synthesized image

        Returns:
            I_refined : (B, 3, H, W) final refined image
            offset_list : list of (B, 3, H, W) COSE offsets
        """

        device = Ix.device
        n = len(prior_list)

        # --------------------------------------------------
        # 1. COSE-based pixel offsets
        # --------------------------------------------------
        
        offset_list: List[torch.Tensor] = []
        for i, P_i in enumerate(prior_list):
            offset_i = self.cose_list[i](Ix, P_i)
            offset_list.append(offset_i.clamp(-1.0, 1.0))  # !!!here clamp offsets to [-1,1]/or [0,1]?

        # --------------------------
        # CAS: confidence-aware weighting
        # --------------------------
        weighted_offsets, cas_weights = self.cas(offset_list)

        # add synthesis prior as an extra offset
        final_offsets = weighted_offsets + [I_synthesis.clamp(0.0, 1.0)]
        # final_offsets = offset_list

        # --------------------------------------------------
        # 2. Pixel-level COMO-Mamba fusion
        # --------------------------------------------------
        # COMO returns Iy = Ix + learned residual internally
        como_out = self.como(final_offsets)
        I_refined = torch.clamp(como_out["Iy"], 0.0, 1.0)

        # new version
        # raw_out = como_out["Iy"]
        # residual = raw_out - I_synthesis
        # I_refined = torch.clamp(I_synthesis + residual, 0, 1)


        self.last_cas_weights = cas_weights

        # --------------------------------------------------
        # Attention entropy statistics (NO GRAD)
        # --------------------------------------------------
        self.last_attn_entropy = como_out.get("att_entropy", None)
        self.last_prior_usage = como_out.get("prior_usage", None)

        return I_refined, weighted_offsets, como_out
    
    def forward_with_offsets(self, offsets):
            como_out = self.como(offsets)
            I_refined = torch.clamp(como_out["Iy"], 0.0, 1.0)

            return {
                "I_refined": I_refined,
                "como_out": como_out
            }