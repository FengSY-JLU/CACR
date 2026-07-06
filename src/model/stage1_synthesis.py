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
            attn_list.append(layer(x))  

        A_sum = sum(attn_list) + 1e-6
        attn_list = [A / A_sum for A in attn_list]

        return attn_list  # List[(B,1,H,W)]


# ----------------------------------------------
# Stage1: SynthesisModule
# ----------------------------------------------
class SynthesisModule(nn.Module):
    def __init__(self, prior_modules):
        """
        prior_modules: List[nn.Module]
        prior_module(x) -> enhanced_image (B,3,H,W)
        """
        assert len(prior_modules) > 0, "SynthesisModule requires >=1 prior module"

        super().__init__()

        self.prior_modules = nn.ModuleList(prior_modules)
        self.num_priors = len(prior_modules)

        self.attn = MultiAttention(num_priors=self.num_priors, in_ch=3)

    def forward(self, x):
        """
        input:
            x:  (B,3,H,W)

        out:
            I_synthesis : (B,3,H,W)
            attn_list   : List[(B,1,H,W)]
            residual_list : List[(B,3,H,W)]
            prior_list  : List[(B,3,H,W)]
        """

        device = x.device  

        prior_list = [module(x) for module in self.prior_modules]
        # e.g. [I_gray, I_udcp, I_lab]

        prior_list = [Pi.to(device) for Pi in prior_list] 

        attn_list = self.attn(x)

        # safer init: zeros tensor with same device/dtype as priors
        I_synthesis = torch.zeros_like(prior_list[0])
        for Ai, Pi in zip(attn_list, prior_list):
            I_synthesis = I_synthesis + Ai * Pi
    

        residual_list = [torch.clamp(Pi - x, -1, 1) for Pi in prior_list]

        return I_synthesis, attn_list, residual_list, prior_list