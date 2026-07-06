# src/model/prior.py
"""
High-quality prior modules for image pre-enhancement.

Assumptions:
 - Inputs: Linear RGB, torch.Tensor, shape (B,3,H,W), values roughly in [0,1]
 - Outputs: Linear RGB, torch.Tensor, shape (B,3,H,W), clamped to [0,1]
 - If OpenCV / skimage are available they will be used for higher-quality ops.
 - Designed for batch processing; conversions to numpy/cv2 are vectorized per-image.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from .restormer_arch import Restormer


# ---------------------------------------------------------
# Color Space Conversion
# ---------------------------------------------------------
def linear_to_srgb(x):

    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * torch.pow(
            x.clamp(min=1e-8),
            1 / 2.4
        ) - 0.055
    )


def srgb_to_linear(x):

    return torch.where(
        x <= 0.04045,
        x / 12.92,
        torch.pow(
            (x + 0.055) / 1.055,
            2.4
        )
    )


# Prefer high-performance libs if available
_HAS_CV2 = False
_HAS_SKIMAGE = False
_HAS_NUMPY = False
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None

try:
    from skimage import exposure, restoration, color, img_as_float32
    _HAS_SKIMAGE = True
except Exception:
    exposure = None
    restoration = None
    color = None

# small utility helpers -----------------------------------------------------
def _tensor_to_uint8_numpy(img_t):
    """
    Convert single image tensor (3,H,W) linear RGB [0,1] to uint8 numpy BGR (H,W,3) for OpenCV.
    Keep in linear domain; cv2 expects 0..255.
    """
    if not _HAS_NUMPY:
        raise RuntimeError("NumPy required for cv2 conversions")
    arr = (img_t.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)  # C,H,W
    arr = arr.transpose(1, 2, 0)  # H,W,C
    # OpenCV often uses BGR order
    bgr = arr[..., ::-1].copy()
    return bgr

def _uint8_bgr_to_tensor(img_bgr):
    """
    img_bgr: numpy uint8 HWC BGR -> returns torch tensor 3,H,W float in [0,1] linear RGB
    """
    if not _HAS_NUMPY:
        raise RuntimeError("NumPy required for cv2 conversions")
    rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).float()
    return t

def _batch_apply_numpy(func, imgs_torch):
    """
    Apply a function that accepts/returns numpy HWC arrays per image.
    imgs_torch: torch.Tensor (B,3,H,W)
    func: callable(np_img_hwc_rgb) -> np_img_hwc_rgb
    Returns: torch.Tensor (B,3,H,W)
    """
    B = imgs_torch.shape[0]
    out_list = []
    for i in range(B):
        img = imgs_torch[i]
        img_np = _tensor_to_uint8_numpy(img)  # uint8 BGR
        out_np = func(img_np)
        if out_np is None:
            out_np = img_np
        # ensure uint8 HWC BGR -> convert to tensor linear RGB
        if out_np.dtype != np.uint8:
            # assume float [0,1] HWC or [0,255] HWC
            if out_np.max() <= 1.0:
                out_uint8 = (out_np * 255.0).clip(0, 255).astype(np.uint8)
            else:
                out_uint8 = out_np.clip(0, 255).astype(np.uint8)
        else:
            out_uint8 = out_np
        t = _uint8_bgr_to_tensor(out_uint8)
        out_list.append(t)
    return torch.stack(out_list, dim=0)

# ---------------------------------------------------------
# Prior Base Class
# ---------------------------------------------------------
class PriorBase(nn.Module):

    color_space = "linear"

    def forward_prior(self, x):
        raise NotImplementedError

    def forward(self, x):

        if self.color_space == "linear":

            return self.forward_prior(x)

        elif self.color_space == "srgb":

            x_srgb = linear_to_srgb(
                x.clamp(0, 1)
            )

            y = self.forward_prior(x_srgb)

            y_linear = srgb_to_linear(
                y.clamp(0, 1)
            )

            return y_linear.clamp(0, 1)

        else:

            raise ValueError(
                f"Unknown color space: "
                f"{self.color_space}"
            )


# -------------------------------------------------------------------------
# 2) CLAHE (Adaptive histogram equalization) on luminance — high-quality
# -------------------------------------------------------------------------
class CLAHEEnhancement(PriorBase):

    color_space = "srgb"
    """
    Apply CLAHE on luminance channel (Lab) — performed per-image via OpenCV if available.
    Works in linear RGB domain by converting to Lab via sRGB path;
    because user handles linear/sRGB externally, we *assume* input is linear RGB:
      - If cv2 available: convert linear RGB -> 8bit sRGB-ish by gamma-encoding approximatedly,
        then work in Lab, apply CLAHE on L, convert back and return linear RGB approx.
    Fallback: use skimage.exposure.equalize_adapthist on per-channel intensity if no cv2.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        super().__init__()
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)

    def forward_prior(self,x):
        B, C, H, W = x.shape
        if _HAS_CV2 and _HAS_NUMPY:
            def op(img_bgr_uint8):
                # img_bgr_uint8: uint8 BGR HWC
                lab = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
                l2 = clahe.apply(l)
                lab2 = cv2.merge([l2, a, b])
                out_bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
                return out_bgr
            return _batch_apply_numpy(op, x)
        else:
            # fallback: per-channel local equalization on intensity (skimage if available)
            out = []
            for i in range(B):
                img = x[i].detach().cpu().numpy().transpose(1,2,0)  # HWC linear
                if _HAS_SKIMAGE:
                    img_float = img_as_float32(img)
                    # convert to HSV-like luminance via rgb2lab/or approximate luminance
                    try:
                        lab = color.rgb2lab(img_float)
                        L = lab[..., 0] / 100.0  # normalize
                        L_eq = exposure.equalize_adapthist(L, clip_limit=self.clip_limit)
                        lab[..., 0] = L_eq * 100.0
                        rgb = color.lab2rgb(lab)
                    except Exception:
                        # fallback to channel-wise CLAHE-like
                        rgb = np.zeros_like(img)
                        for c in range(3):
                            rgb[..., c] = exposure.equalize_adapthist(img[..., c], clip_limit=self.clip_limit)
                    rgb = np.clip(rgb, 0.0, 1.0)
                    t = torch.from_numpy(rgb.transpose(2,0,1)).float()
                else:
                    # naive fallback: per-channel histogram equalization via torch (cheap)
                    t = x[i]
                    for c in range(3):
                        ch = t[c:c+1]
                        # small local contrast stretch
                        lo = ch.quantile(0.02)
                        hi = ch.quantile(0.98)
                        t[c:c+1] = (ch - lo) / (hi - lo + 1e-6)
                    t = torch.clamp(t, 0.0, 1.0)
                out.append(t)
            return torch.stack(out, dim=0)


# -------------------------------------------------------------------------
# 3) Improved UDCP (uses local min via cv2 if available)
# -------------------------------------------------------------------------
class UDCPColorRestorationHighQuality(PriorBase):

    color_space = "linear"
    """
    Improved UDCP / UWCID style restoration:
     - Uses morphological local-min (dark channel style) by cv2.erode for speed
     - Robust A estimation via top-percentile pixels
     - Works on linear RGB input; returns linear RGB
    """
    def __init__(self, omega=0.8, t0=0.05, kernel_size=15, top_percent=0.001):
        super().__init__()
        self.omega = float(omega)
        self.t0 = float(t0)
        self.k = int(kernel_size) if kernel_size % 2 == 1 else int(kernel_size)+1
        self.top_percent = float(top_percent)

    def _local_min_np(self, channel_np, k):
        # channel_np: HWC or single channel HxW in uint8 or float
        # perform min filter via erosion in cv2
        if _HAS_CV2:
            if channel_np.dtype != np.uint8:
                # assume float in [0,1] -> convert to 0..255
                ch = (channel_np * 255.0).clip(0,255).astype(np.uint8)
            else:
                ch = channel_np
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
            min_ch = cv2.erode(ch, kernel)
            if channel_np.dtype != np.uint8:
                min_ch = min_ch.astype(np.float32) / 255.0
            return min_ch
        else:
            # pure numpy fallback: sliding window min via stride tricks (slow)
            from skimage.morphology import erosion, square
            if channel_np.max() <= 1.0:
                chf = (channel_np * 255.0).astype(np.uint8)
                min_ch = erosion(chf, square(k)).astype(np.float32) / 255.0
            else:
                min_ch = erosion(channel_np.astype(np.uint8), square(k)).astype(np.float32) / 255.0
            return min_ch

    def forward_prior(self,x):
        # x: torch (B,3,H,W)
        B, C, H, W = x.shape
        out_list = []
        for i in range(B):
            img = x[i].detach().cpu().numpy().transpose(1,2,0)  # HWC linear float
            # operate per-channel with uint8 conversion for cv2 ops
            # compute local min on green and blue channels using erosion
            g = img[..., 1]
            b = img[..., 2]
            # use _local_min_np on float arrays
            min_g = self._local_min_np(g, self.k)
            min_b = self._local_min_np(b, self.k)
            min_gb = np.minimum(min_g, min_b)
            # transmission estimate
            t = 1.0 - self.omega * min_gb
            t = np.clip(t, self.t0, 1.0)
            # robust A: top-percentile on each channel
            flat = img.reshape(-1,3)
            k = max(1, int(flat.shape[0] * self.top_percent))
            A = np.zeros(3, dtype=np.float32)
            for ch in range(3):
                vals = np.partition(flat[:,ch], -k)[-k:]
                A[ch] = vals.mean()
            A = A.reshape(1,1,3)  # HWC broadcast
            # recover J = (I - A) / t + A  (broadcast properly)
            denom = t[..., None]
            J = (img - A) / (denom + 1e-8) + A
            J = np.clip(J, 0.0, 1.0)
            t_t = torch.from_numpy(J.transpose(2,0,1)).float()
            out_list.append(t_t)
        return torch.stack(out_list, dim=0)


# -------------------------------------------------------------------------
# 4) MSRCR Retinex (Multi-scale Retinex with Color Restoration) — high-quality
# -------------------------------------------------------------------------
class MSRCRRetinex(PriorBase):

    color_space = "srgb"
    """
    Multi-scale Retinex with Color Restoration (MSRCR).
    Implementation follows classical formulas; heavy but gives strong visual enhancement.
    Input: linear RGB [0,1]; we do computations in float32; returns linear RGB clipped [0,1].
    """
    def __init__(self, scales=(15, 80, 250), G=5.0, b=25.0, alpha=125.0, beta=46.0):
        super().__init__()
        self.scales = tuple(scales)
        self.G = float(G)
        self.b = float(b)
        self.alpha = float(alpha)
        self.beta = float(beta)

    @staticmethod
    def _single_scale_retinex(img, sigma):
        # img: HxWx3 numpy float (0..1)
        import cv2
        blur = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=sigma, sigmaY=sigma)
        # avoid log(0)
        eps = 1e-6
        ret = np.log(img + eps) - np.log(blur + eps)
        return ret

    def forward_prior(self,x):
        B, C, H, W = x.shape
        out_batch = []
        use_cv = _HAS_CV2 and _HAS_NUMPY
        for i in range(B):
            img = x[i].detach().cpu().numpy().transpose(1,2,0).astype(np.float32)  # HWC
            if use_cv:
                msr = np.zeros_like(img)
                for s in self.scales:
                    msr += self._single_scale_retinex(img, s)
                msr = msr / float(len(self.scales))
                # color restoration function
                # crf = self.beta * (np.log(self.alpha * img + 1.0))
                # simplified CRF:
                intensity = img.sum(axis=2, keepdims=True) + 1e-6
                crf = self.beta * (np.log(self.alpha * img + 1.0) - np.log(intensity + 1e-6))
                msrcr = self.G * (msr * crf + self.b)
                # normalize to 0..1 using percentile stretch
                lo = np.percentile(msrcr, 1)
                hi = np.percentile(msrcr, 99)
                msrcr = (msrcr - lo) / (hi - lo + 1e-8)
                msrcr = np.clip(msrcr, 0.0, 1.0)
                out = msrcr
            else:
                # fallback: simple SSR with a single small sigma via torch gaussian blur (approx)
                t = x[i:i+1]
                # use illumination estimate as gaussian blur by avg_pool
                L = t.mean(dim=1, keepdim=True)
                L_blur = F.avg_pool2d(L, kernel_size=15, stride=1, padding=7)
                ret = torch.log(t + 1e-6) - torch.log(torch.cat([L_blur,L_blur,L_blur], dim=1) + 1e-6)
                out_t = ret.squeeze(0).detach().cpu().numpy()
                lo = np.percentile(out_t, 1)
                hi = np.percentile(out_t, 99)
                out = np.clip((out_t - lo) / (hi - lo + 1e-8), 0.0, 1.0).transpose(1,2,0)
            out_batch.append(torch.from_numpy(out.transpose(2,0,1)).float())
        return torch.stack(out_batch, dim=0)

# -------------------------------------------------------------------------
# Optional: Unsharp Masking Prior (simple edge enhancement)
# -------------------------------------------------------------------------
class UnsharpMask(PriorBase):

    color_space = "srgb"

    def __init__(self, amount=1.5, kernel=5):
        super().__init__()

        self.amount = amount
        self.kernel = kernel

    def forward_prior(self,x):

        blur = F.avg_pool2d(
            x,
            kernel_size=self.kernel,
            stride=1,
            padding=self.kernel//2
        )

        sharp = x + self.amount * (x - blur)

        return torch.clamp(sharp,0,1)


def gray_world_adjust(img):
    
    mean_rgb = img.mean(dim=[2,3], keepdim=True)
    mean_gray = mean_rgb.mean(dim=1, keepdim=True)
    scale = mean_gray / (mean_rgb + 1e-8)
    out = img * scale
    return torch.clamp(out, 0, 1)


class GrayWorldCompensator(PriorBase):
    color_space = "linear"
    def __init__(self):
        super().__init__()
    def forward_prior(self,x):
        return gray_world_adjust(x)


class HDPPrior(PriorBase):

    color_space = "linear"
    """
    Histogram Distribution Prior (HDP)

    Reproduced from:

    Underwater Image Enhancement by Dehazing
    With Minimum Information Loss and
    Histogram Distribution Prior

    We implement the core idea:

        Histogram Distribution Matching

    rather than simple histogram stretching.

    Input:
        (B,3,H,W) linear RGB

    Output:
        (B,3,H,W)
    """
    def __init__(self, bins=256):
        super().__init__()
        self.bins = bins

    def _cdf_match(self, img):

        flat = img.flatten()

        hist = torch.histc(flat, bins=self.bins, min=0.0, max=1.0)

        hist = hist / (hist.sum() + 1e-8)

        src_cdf = torch.cumsum(hist, dim=0)

        #
        # HDP target distribution
        #
        # Natural-image-like bell shape
        #

        x = torch.linspace(0, 1, self.bins, device=img.device)

        target_hist = torch.exp(
            -((x - 0.5) ** 2) / (2 * 0.18 ** 2)
        )

        target_hist = target_hist / target_hist.sum()

        tgt_cdf = torch.cumsum(
            target_hist,
            dim=0
        )

        #
        # Build LUT
        #

        lut = torch.zeros(self.bins, device=img.device)

        for i in range(self.bins):

            diff = torch.abs(
                src_cdf[i] - tgt_cdf
            )

            lut[i] = torch.argmin(diff)

        lut = lut / (self.bins - 1)

        idx = (
            img * (self.bins - 1)
        ).long().clamp(
            0,
            self.bins - 1
        )

        out = lut[idx]

        return out

    def forward_prior(self,x):

        B,C,H,W = x.shape

        outputs = []

        for b in range(B):

            img = x[b]

            #
            # luminance
            #

            R = img[0:1]
            G = img[1:2]
            Bc = img[2:3]

            Y = (0.299 * R + 0.587 * G + 0.114 * Bc)

            U = Bc - Y
            V = R - Y

            Y_new = self._cdf_match(Y)

            R_new = Y_new + V
            B_new = Y_new + U

            G_new = (Y_new - 0.299 * R_new - 0.114 * B_new) / 0.587

            out = torch.cat([R_new, G_new, B_new], dim=0)

            outputs.append(torch.clamp(out,0,1))

        return torch.stack(outputs)


class LearnedPrior(PriorBase):

    color_space = "srgb"

    def __init__(self,
                 model,
                 color_space="srgb",
                 name=None):

        super().__init__()

        self.model = model.eval()

        self.color_space = color_space

        self.name = name

    @torch.no_grad()
    def forward_prior(self, x):

        B,C,H,W = x.shape

        factor = 8

        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor

        if pad_h > 0 or pad_w > 0:

            x = F.pad(
                x,
                (0,pad_w,0,pad_h),
                mode="reflect"
            )

        out = self.model(x)

        out = out[:, :, :H, :W]

        return torch.clamp(out,0,1)

def build_restormer_prior(
        model_path,
        device
    ):

    model = Restormer(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4,6,6,8],
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=False
    ).to(device)

    state = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(state)

    model.eval()

    return LearnedPrior(
        model=model,
        color_space="srgb",
        name=os.path.basename(model_path)
    )

# -------------------------------------------------------------------------
# Utility: build a default prior module list
# -------------------------------------------------------------------------
def default_priors(device: torch.device, learned_prior_cfgs: str = None):
   
    priors = [
        UDCPColorRestorationHighQuality(omega=0.8, t0=0.05, kernel_size=15, top_percent=0.001),
        CLAHEEnhancement(clip_limit=2.0, tile_grid_size=(8,8)),
        MSRCRRetinex(scales=(15,80,250)),
        UnsharpMask(amount=1.5, kernel=5),
        HDPPrior(),
        GrayWorldCompensator()
    ]
    
    if learned_prior_cfgs is not None:

        for cfg in learned_prior_cfgs:

            if cfg["type"] == "restormer":

                priors.append(

                    build_restormer_prior(
                        cfg["path"],
                        device
                    )
                )

                print(
                    f"[Info] Loaded Restormer prior:"
                    f"{cfg['path']}"
                )

    return priors

# End of file
