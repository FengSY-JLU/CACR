# src/model/dconv/cose_stable.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorShiftEstimation(nn.Module):
    def __init__(self, in_channels=3, kernel_size=3, modulation=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.stride = 1
        self.modulation = modulation

        self.channel_down = nn.Conv2d(in_channels * 2, in_channels, 1)

        N = kernel_size * kernel_size

        self.offset_conv = nn.Conv2d(in_channels, 2 * N, 3, padding=1)

        if modulation:
            self.modulation_conv = nn.Conv2d(in_channels, N, 3, padding=1)

        self.color_offset_conv = nn.Conv2d(in_channels, in_channels * N, 3, padding=1)

        self.final_conv = nn.Conv2d(in_channels, in_channels,
                                    kernel_size=kernel_size,
                                    stride=kernel_size)

        self.zero_padding = nn.ZeroPad2d(self.padding)

    def forward(self, x, ref):

        B, C, H, W = x.shape
        N = self.kernel_size * self.kernel_size

        fused = torch.cat([x, ref], dim=1)
        fused = self.channel_down(fused)

        # ----------------------------------
        # 1. OFFSET (bounded)
        # ----------------------------------
        offset = torch.tanh(self.offset_conv(fused)) * (self.kernel_size / 2)

        # ----------------------------------
        # 2. MODULATION (safe)
        # ----------------------------------
        if self.modulation:
            m = torch.sigmoid(self.modulation_conv(fused))
        else:
            m = None

        # ----------------------------------
        # 3. COLOR OFFSET (scaled residual)
        # ----------------------------------
        c_offset = 0.5 * torch.tanh(self.color_offset_conv(fused))
        c_offset = c_offset.view(B, C, N, H, W).permute(0, 1, 3, 4, 2)

        # ----------------------------------
        # 4. SAMPLING
        # ----------------------------------
        x_pad = self.zero_padding(x)

        p = self._get_sampling_locations(offset, x_pad, H, W)

        # detach floor
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = self._clip(q_lt, x_pad)
        q_rb = self._clip(q_rb, x_pad)

        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        p = self._clip(p, x_pad)

        # ----------------------------------
        # 5. BILINEAR (stable)
        # ----------------------------------
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * \
               (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))

        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * \
               (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))

        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * \
               (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))

        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * \
               (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        g_lt = torch.tanh(g_lt) * 2.0
        g_rb = torch.tanh(g_rb) * 2.0
        g_lb = torch.tanh(g_lb) * 2.0
        g_rt = torch.tanh(g_rt) * 2.0

        x_q_lt = self._gather(x_pad, q_lt, N, H, W)
        x_q_rb = self._gather(x_pad, q_rb, N, H, W)
        x_q_lb = self._gather(x_pad, q_lb, N, H, W)
        x_q_rt = self._gather(x_pad, q_rt, N, H, W)

        x_offset = (
            g_lt.unsqueeze(1) * x_q_lt +
            g_rb.unsqueeze(1) * x_q_rb +
            g_lb.unsqueeze(1) * x_q_lb +
            g_rt.unsqueeze(1) * x_q_rt
        )

        # ----------------------------------
        # 6. COLOR + MODULATION
        # ----------------------------------
        x_offset = x_offset + c_offset

        if m is not None:
            m = m.permute(0, 2, 3, 1).unsqueeze(1).repeat(1, C, 1, 1, 1)
            x_offset = x_offset * m

        # ----------------------------------
        # 7. RESHAPE + CONV
        # ----------------------------------
        x_offset = self._reshape(x_offset)

        out = self.final_conv(x_offset)

        out = torch.nan_to_num(out, 0.0, 0.0, 0.0)

        return out

    # ==========================================================
    # Helper functions
    # ==========================================================

    def _get_sampling_locations(self, offset, x, H, W):
        B, _, H, W = offset.shape
        N = offset.shape[1] // 2
        device = offset.device
        dtype = offset.dtype

        p_n = self._get_p_n(N, dtype, device)
        p_0 = self._get_p_0(H, W, N, dtype, device)

        return (p_0 + p_n + offset).permute(0, 2, 3, 1)

    def _get_p_n(self, N, dtype, device):
        ks = self.kernel_size
        y, x = torch.meshgrid(
            torch.arange(-(ks // 2), ks // 2 + 1, device=device),
            torch.arange(-(ks // 2), ks // 2 + 1, device=device),
            indexing='ij'
        )
        p_n = torch.cat([y.flatten(), x.flatten()], dim=0).view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, H, W, N, dtype, device):
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        p0_y = y.unsqueeze(0).unsqueeze(0).repeat(1, N, 1, 1)
        p0_x = x.unsqueeze(0).unsqueeze(0).repeat(1, N, 1, 1)
        return torch.cat([p0_y, p0_x], dim=1).type(dtype)

    def _clip(self, p, x):
        N = self.kernel_size * self.kernel_size
        return torch.cat([
            torch.clamp(p[..., :N], 0, x.shape[2] - 1),
            torch.clamp(p[..., N:], 0, x.shape[3] - 1)
        ], dim=-1).long()

    def _gather(self, x, q, N, H, W):
        B, C, Hp, Wp = x.shape
        x_flat = x.view(B, C, -1)
        idx = q[..., :N] * Wp + q[..., N:]
        idx = idx.unsqueeze(1).expand(-1, C, -1, -1, -1).reshape(B, C, -1)
        return x_flat.gather(-1, idx).view(B, C, H, W, N)

    def _reshape(self, x_offset):
        B, C, H, W, N = x_offset.size()
        ks = self.kernel_size
        x_offset = torch.cat([
            x_offset[..., s:s + ks].contiguous().view(B, C, H, W * ks)
            for s in range(0, N, ks)
        ], dim=-1)
        return x_offset.view(B, C, H * ks, W * ks)
