import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class OffsetCAS(nn.Module):
    """
    Confidence-Aware Selection (CAS) for COSE offsets.

    Input : list of COSE offsets Δ_i = COSE(Ix, P_i)
    Output: per-pixel confidence-weighted offsets
    """

    def __init__(self, offset_ch=3, hidden=32):
        super().__init__()

        # Shared encoder to ensure fair comparison across priors
        self.encoder = nn.Sequential(
            nn.Conv2d(offset_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Confidence score head
        self.score_head = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, offset_list: List[torch.Tensor]):
        """
        Args:
            offset_list: list of (B, 3, H, W), COSE outputs
        Returns:
            weighted_offsets: list of (B, 3, H, W)
            weights: (B, N, H, W)
        """
        scores = []

        for offset in offset_list:
            feat = self.encoder(offset)
            score = self.score_head(feat)   # (B,1,H,W)
            scores.append(score)

        # Stack and normalize
        score_map = torch.cat(scores, dim=1)     # (B,N,H,W)
        weights = F.softmax(score_map, dim=1)

        # Apply weights
        weighted_offsets = [
            weights[:, i:i+1] * offset_list[i]
            for i in range(len(offset_list))
        ]

        return weighted_offsets, weights

