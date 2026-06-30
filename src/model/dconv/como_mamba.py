# src/model/dconv/como_mamba.py
# --------------------------------------------------------------
# COMO-Mamba Hybrid v3.1 (Anchor-free / Prior-only Fusion)
#
# Design philosophy:
#   - NO explicit Ix anchor
#   - Input = multiple pixel-level hypotheses (priors / offsets)
#   - Output = fused enhanced image Iy
#
# Core components:
#   - Multi-directional Mamba (sequence modeling)
#   - Direction attention (softmax over K scan directions)
#   - Depthwise MLP-Mixer (local refinement)
#   - Spatial gating for hybrid fusion
# --------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------
# Mamba import (with safe fallback)
# --------------------------------------------------------------
try:
    from mamba_ssm import Mamba
    MAMBA_OK = True
except Exception:
    try:
        from PseudoMamba import Mamba
        MAMBA_OK = True
    except Exception:
        MAMBA_OK = False

def compute_attention_entropy(att, eps=1e-8):
    """
    att: (B, K, C, H, W) softmax-normalized
    Returns:
        entropy_map: (B, C, H, W)
        mean_entropy: scalar tensor
        prior_usage: (K,) averaged attention weight
    """
    # entropy per pixel & channel
    entropy = - (att * torch.log(att + eps)).sum(dim=1)  # sum over K
    # entropy: (B, C, H, W)

    # global mean entropy
    mean_entropy = entropy.mean()

    # prior usage statistics
    # average over batch, channel, spatial dims
    prior_usage = att.mean(dim=(0, 2, 3, 4))  # (K,)

    return entropy, mean_entropy, prior_usage

def soft_prune_attention(att, lambda_prune=0.05, eps=1e-6):
    """
    att: (B, K, C, H, W)
    """
    # global usage per direction
    mean_attn = att.mean(dim=(0, 2, 3, 4))  # (K,)

    mask = (mean_attn > lambda_prune).float().view(1, -1, 1, 1, 1)

    att = att * mask
    att = att / (att.sum(dim=1, keepdim=True) + eps)
    return att


# --------------------------------------------------------------
# Fallback sequence mixer (if Mamba unavailable)
# --------------------------------------------------------------
class SeqMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        # x: (B, N, C)
        return x + self.block(x)


# --------------------------------------------------------------
# Unified Mamba wrapper
# --------------------------------------------------------------
class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if MAMBA_OK:
            try:
                try:
                    self.m = Mamba(d_model=dim)
                except TypeError:
                    self.m = Mamba(dim)
                self.use_mamba = True
            except Exception:
                self.m = SeqMixer(dim)
                self.use_mamba = False
        else:
            self.m = SeqMixer(dim)
            self.use_mamba = False

    def forward(self, x):
        # x: (B, N, C)
        if self.use_mamba:
            try:
                return self.m(x)
            except Exception:
                try:
                    return self.m(x.transpose(1, 0)).transpose(1, 0)
                except Exception:
                    return self.m(x)
        else:
            return self.m(x)


# --------------------------------------------------------------
# Depthwise MLP-Mixer (local branch)
# --------------------------------------------------------------
class DWMLPMixer(nn.Module):
    """
    Depthwise Conv → Channel MLP → residual
    Preserves local structure while enabling channel mixing.
    """
    def __init__(self, dim, kernel=3, mlp_ratio=2):
        super().__init__()
        pad = kernel // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel, padding=pad, groups=dim)
        self.act = nn.GELU()
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim * mlp_ratio, dim, 1),
        )

    def forward(self, x):
        out = self.dwconv(x)
        out = self.act(out)
        out = self.channel_mlp(out)
        return out + x


# --------------------------------------------------------------
# Main COMO-Mamba (Anchor-free)
# --------------------------------------------------------------
class COMO_Mamba(nn.Module):
    """
    COMO-Mamba v3.1 (Anchor-free Fusion)

    Input:
        offsets: List[(B, 3, H, W)]
            - color hypotheses / priors
            - e.g. [offset1, offset2, ..., I_synthesis]

    Output:
        Iy: (B, 3, H, W)
            - fused enhanced image
    """

    def __init__(
        self,
        offset_ch=3,
        dim=64,
        n_offsets=4,
        use_space2depth=True,
        factor=2,
        use_multidirectional=True,
        md_directions=4,
    ):
        super().__init__()

        self.offset_ch = offset_ch
        self.dim = dim
        self.use_s2d = use_space2depth
        self.s2d_factor = factor

        self.use_multidirectional = use_multidirectional
        self.md_directions = md_directions if use_multidirectional else 1

        # built dynamically based on number of priors
        self.n_offsets = n_offsets
        total_in = n_offsets * offset_ch
        self.fuse_conv = nn.Conv2d(offset_ch, dim, 1)

        # positional encoding (x, y)
        self.pos_proj = nn.Sequential(
            nn.Conv2d(2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

        # space-to-depth
        if self.use_s2d:
            self.s2d = nn.PixelUnshuffle(factor)
            self.reduce = nn.Conv2d(dim * factor * factor, dim, 1)
        else:
            self.s2d = None

        # Mamba blocks (per direction)
        self.mamba_blocks = nn.ModuleList(
            [MambaBlock(dim) for _ in range(self.md_directions)]
        )

        # local refinement
        self.dwmlp = DWMLPMixer(dim)

        # spatial gating
        self.gating_conv = nn.Conv2d(dim, 1, kernel_size=1)

        # direction attention
        self.dir_att = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, self.md_directions * dim, 1),
        )

        # mixing projection
        self.mix_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

        # ----------------------------------------------------------
        # Offset Attention
        # ----------------------------------------------------------
        self.offset_score = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1)
        )

        # output head
        self.out = nn.Conv2d(dim, 3, kernel_size=1)

    # ----------------------------------------------------------
    # Scan index helpers
    # ----------------------------------------------------------
    def _build_scan_indices(self, H, W, device):
        N = H * W
        base = torch.arange(N, device=device).view(H, W)
        idxs = [
            base.reshape(-1),
            torch.flip(base, (0, 1)).reshape(-1),
            base.t().reshape(-1),
            torch.flip(base.t(), (0, 1)).reshape(-1),
        ]
        return [idxs[i].long() for i in range(self.md_directions)]

    def _flatten_with_index(self, x, idx):
        B, C, H, W = x.shape
        N = H * W
        xf = x.view(B, C, N)
        idx_exp = idx.view(1, 1, N).expand(B, C, N)
        out = torch.gather(xf, 2, idx_exp)
        return out.permute(0, 2, 1).contiguous()  # (B, N, C)

    def _scatter_to_image(self, seq, idx, H, W):
        B, N, C = seq.shape
        seq_t = seq.permute(0, 2, 1).contiguous()
        out = torch.zeros(B, C, N, device=seq.device, dtype=seq.dtype)
        idx_exp = idx.view(1, 1, N).expand(B, C, N)
        out.scatter_(2, idx_exp, seq_t)
        return out.view(B, C, H, W)

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------
    def forward(self, offsets):
        """
        offsets: list of (B, 3, H, W)
        """
        assert isinstance(offsets, (list, tuple)) and len(offsets) > 0

        B, _, H, W = offsets[0].shape
        device = offsets[0].device
        n_offsets = len(offsets)

        assert n_offsets == self.n_offsets, \
            f"Expected {self.n_offsets} offsets, got {n_offsets}"
        
        # ----------------------------------------------------------
        # 1) Offset Attention Fusion
        # ----------------------------------------------------------

        # ----------------------------------------------------------
        # ALL contributors attention
        # (used for manifold regularization)
        # ----------------------------------------------------------

        # ----------------------------------------------------------
        # expert routing
        # synthesis excluded
        # ----------------------------------------------------------

        expert_offsets = offsets[:-1]

        expert_scores = []

        for off in expert_offsets:

            s = self.offset_score(off)

            expert_scores.append(s)

        expert_scores = torch.stack(
            expert_scores,
            dim=1
        )

        # (B,K,1,H,W)

        att_offset_only = F.softmax(
            expert_scores,
            dim=1
        )

        # ----------------------------------------------------------
        # weighted expert fusion
        # ----------------------------------------------------------

        expert_stack = torch.stack(
            expert_offsets,
            dim=1
        )

        fused_expert = (
            att_offset_only * expert_stack
        ).sum(dim=1)

        # ----------------------------------------------------------
        # synthesis hypothesis
        # ----------------------------------------------------------

        I_synthesis = offsets[-1]

        # synthesis participates in hypothesis fusion
        # but NOT expert competition
        # ----------------------------------------------------------

        fused_offset = fused_expert + I_synthesis

        # ----------------------------------------------------------
        # OFFSET-ONLY attention
        # (exclude synthesis)
        # used for contributor competition
        # ----------------------------------------------------------

        # ----------------------------------------------------------
        # 2) projection
        # ----------------------------------------------------------
        x = self.fuse_conv(fused_offset)

        # 2) positional encoding
        row = torch.linspace(-1, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        col = torch.linspace(-1, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        pos = torch.cat([row, col], dim=1)
        x = x + self.pos_proj(pos)

        # 3) space-to-depth
        if self.s2d is not None:
            x = self.s2d(x)
            x = self.reduce(x)
            Hp, Wp = H // self.s2d_factor, W // self.s2d_factor
        else:
            Hp, Wp = H, W

        # 4) multi-directional Mamba
        idx_list = self._build_scan_indices(Hp, Wp, device)
        recon_list = []
        for i, idx in enumerate(idx_list):
            seq = self._flatten_with_index(x, idx)
            seq_out = self.mamba_blocks[i](seq)
            rec = self._scatter_to_image(seq_out, idx, Hp, Wp)
            recon_list.append(rec)

        recon_stack = torch.stack(recon_list, dim=1)  # (B, K, C, H, W)

        # 5) direction attention
        recon_avg = recon_stack.mean(dim=1)
        att_raw = self.dir_att(recon_avg)
        att = att_raw.view(B, self.md_directions, self.dim, Hp, Wp)
        att = F.softmax(att, dim=1)

        # -------------------------------
        # Soft-pruning (training only)
        # -------------------------------
        if self.training:
            # 你可以把 lambda_prune 设成成员变量或外部传入
            lambda_prune = getattr(self, "lambda_prune", 0.05)
            att = soft_prune_attention(att, lambda_prune)

        # --------------------------------------------------
        # Attention entropy statistics (NO gradient)
        # --------------------------------------------------
        with torch.no_grad():
            entropy_map, mean_entropy, prior_usage = compute_attention_entropy(att)

        recon = (att * recon_stack).sum(dim=1)

        # 6) hybrid fusion
        # clip to prevent early epoch collapse
        dw_out = self.dwmlp(recon)
        lambda_map = torch.sigmoid(self.gating_conv(recon))
        dw_out = torch.clamp(dw_out, -0.5, 0.5)   # 可根据数值适当调整范围
        hybrid = lambda_map * recon + (1.0 - lambda_map) * dw_out

        # 7) projection + upsample
        fused = self.mix_proj(hybrid)
        if self.s2d is not None and (Hp != H or Wp != W):
            fused = F.interpolate(fused, size=(H, W), mode="bilinear", align_corners=False)

        # 8) output
        # Soft clamp
        # Iy = torch.tanh(self.out(fused)) * 0.5 + 0.5
        Iy = torch.sigmoid(self.out(fused))

        return {
            "Iy": Iy,

            # offset-only competition
            "att_offset_only": att_offset_only,

            # direction attention
            "att_direction": att,

            "recon_list": recon_list,
            "recon": recon,
            "lambda_map": lambda_map,
            "dw_out": dw_out,

            # ---- attention statistics (for logging / analysis) ----
            "att_entropy": mean_entropy,      # scalar
            "att_entropy_map": entropy_map,    # (B, C, H, W)
            "prior_usage": prior_usage,        # (K,)
        }

