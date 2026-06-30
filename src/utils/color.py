# src/utils/color.py
# safely convert between sRGB and linear RGB (inverse EOTF / EOTF)

import torch


# ============================================================
# sRGB → Linear RGB
# ============================================================
def srgb_to_linear(img):
    """
    img: (B,3,H,W) in [0,1] sRGB
    return: linear RGB in [0,1]
    """
    img = torch.clamp(img, 0.0, 1.0)
    mask = (img > 0.04045).float()
    linear = ((img + 0.055) / 1.055) ** 2.4 * mask + (img / 12.92) * (1 - mask)
    return linear


# ============================================================
# Linear RGB → sRGB
# ============================================================
def linear_to_srgb(img):
    """
    img: (B,3,H,W) linear RGB
    return: sRGB in [0,1]
    """
    img = torch.clamp(img, 0.0, 1.0)
    mask = (img > 0.0031308).float()
    srgb = (1.055 * img.pow(1.0 / 2.4) - 0.055) * mask + (12.92 * img) * (1 - mask)
    return torch.clamp(srgb, 0.0, 1.0)


# ============================================================
# numpy <-> tensor (for evaluation)
# ============================================================
def tensor_to_image(t):
    """
    tensor: (1,3,H,W) or (3,H,W)
    return numpy uint8 HWC
    """
    if t.ndim == 4:
        t = t[0]
    t = torch.clamp(t, 0, 1)
    t = t.permute(1, 2, 0).detach().cpu().numpy()
    t = (t * 255).round().astype("uint8")
    return t


def image_to_tensor(img):
    """
    img: numpy HWC uint8
    return: (3,H,W) float tensor in [0,1] sRGB
    """
    import numpy as np
    if img.dtype != np.uint8:
        raise ValueError("image_to_tensor expects uint8 input")
    t = torch.from_numpy(img.astype("float32") / 255.0)
    t = t.permute(2, 0, 1)
    return t
