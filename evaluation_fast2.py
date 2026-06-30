# evaluation_fast.py
# ============================================================
# Fast Evaluation Script (Training / Ablation)
#   - Refined + Synthesis metrics
#   - Structure gain statistics (benchmark-free)
#   - UIQM / UCIQE comparison
#   - Attention entropy monitoring
#   - NEW: Low / High Frequency Chroma Diagnostics (Lab-a)
#   - SAFE VERSION (NO SIDE EFFECTS)
# ============================================================

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.filters import sobel
from skimage.color import rgb2gray

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# sRGB (UNCHANGED)
# ------------------------------------------------------------
try:
    from src.utils.color import linear_to_srgb
except ImportError:
    def linear_to_srgb(x):
        x = torch.clamp(x, 0.0, 1.0)
        mask = (x > 0.0031308).float()
        return torch.clamp(
            (1.055 * x.pow(1.0 / 2.4) - 0.055) * mask
            + 12.92 * x * (1 - mask),
            0.0, 1.0
        )

# ------------------------------------------------------------
# Underwater metrics
# ------------------------------------------------------------
from src.measurement.metrics_utils import calculate_uciqe, calculate_uiqm


# ------------------------------------------------------------
# Structure energy (benchmark-free)
# ------------------------------------------------------------
def structure_energy(img_chw):
    img = img_chw.detach().cpu().numpy().transpose(1, 2, 0)
    gray = rgb2gray(img)
    edge = sobel(gray)
    return np.mean(edge ** 2)


# ------------------------------------------------------------
# RGB -> Lab (EVAL ONLY, NO IN-PLACE)
# ------------------------------------------------------------
def rgb_to_lab_eval(img_chw):
    """
    img_chw: (3,H,W) in [0,1]
    return: (3,H,W) Lab
    """
    img = img_chw.unsqueeze(0)

    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    xyz = torch.cat([x, y, z], dim=1)
    xyz_ref = torch.tensor(
        [0.95047, 1.00000, 1.08883],
        device=xyz.device
    ).view(1, 3, 1, 1)

    xyz = xyz / (xyz_ref + 1e-12)

    eps = 0.008856
    mask = (xyz > eps).float()
    f = xyz.pow(1.0 / 3.0) * mask + (7.787 * xyz + 16.0 / 116.0) * (1 - mask)

    L = 116 * f[:, 1:2] - 16
    a = 500 * (f[:, 0:1] - f[:, 1:2])
    b = 200 * (f[:, 1:2] - f[:, 2:3])

    return torch.cat([L, a, b], dim=1).squeeze(0)


# ------------------------------------------------------------
# SAFE visualization for Lab-a (NO BLACK IMAGE)
# ------------------------------------------------------------
def vis_chroma_safe(x, clip=40.0):
    """
    x: (H,W) or (1,H,W) Lab-a channel
    return: (3,H,W) in [0,1] for save_image
    """
    x = x.detach().cpu()
    if x.dim() == 3:
        x = x.squeeze(0)

    x = torch.clamp(x, -clip, clip)
    x = (x + clip) / (2 * clip)
    x = x.unsqueeze(0).repeat(3, 1, 1)
    return x


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, opt, dataloader, epoch=None):
    model.eval()
    ep = str(epoch) if epoch is not None else "N/A"
    print(f"\n[Fast Evaluation] Epoch {ep}")

    base_dir = f"{opt.output_folder.rstrip('/')}_{opt.indicator}"
    epoch_dir = os.path.join(base_dir, f"epoch_{ep}")
    os.makedirs(epoch_dir, exist_ok=True)

    acc = {
        "R_PSNR": [], "R_SSIM": [], "R_UCIQE": [], "R_UIQM": [],
        "S_PSNR": [], "S_SSIM": [], "S_UCIQE": [], "S_UIQM": [],
        "AttnEntropy": [],
        "StructGain": [],
    }

    struct_gain_win = 0
    perceptual_win = 0
    total = 0

    pbar = tqdm(dataloader, ncols=140, desc=f"Eval | Epoch {ep}", leave=False)

    for idx, batch in enumerate(pbar):
        x, gt, name = batch[0].to(device), batch[1].to(device), batch[2]
        img_name = name[0]

        outputs = model(x=x, F_DM_l=None)
        I_ref = outputs["I_refined"]
        I_syn = outputs["I_synthesis"]

        # ----------------------------------------------------
        # Save visual diagnostics (UNCHANGED + SAFE ADD-ON)
        # ----------------------------------------------------
        if idx < 3:
            img_dir = os.path.join(epoch_dir, img_name)
            os.makedirs(img_dir, exist_ok=True)

            # --- original behavior (DO NOT TOUCH) ---
            save_image(linear_to_srgb(I_ref), os.path.join(img_dir, "refined.png"))
            save_image(linear_to_srgb(I_syn), os.path.join(img_dir, "synthesis.png"))

            for i, p in enumerate(outputs["priors"]):
                save_image(
                    linear_to_srgb(p.clamp(0, 1)),
                    os.path.join(img_dir, f"prior_{i+1}.png"),
                )

            for i, o in enumerate(outputs["offsets_stage2"]):
                save_image(
                    linear_to_srgb(o.clamp(0, 1)),
                    os.path.join(img_dir, f"offset_{i+1}.png"),
                )

            # ---------- Low / High frequency chroma (SAFE) ----------
            ref_chw = I_ref.squeeze(0).clamp(0, 1)
            syn_chw = I_syn.squeeze(0).clamp(0, 1)

            lab_ref = rgb_to_lab_eval(ref_chw)
            lab_syn = rgb_to_lab_eval(syn_chw)

            a_ref = lab_ref[1]  # (H,W)
            a_syn = lab_syn[1]

            k = 15
            a_ref_l = F.avg_pool2d(a_ref[None, None], k, 1, k // 2).squeeze()
            a_syn_l = F.avg_pool2d(a_syn[None, None], k, 1, k // 2).squeeze()
            a_ref_h = a_ref - a_ref_l

            save_image(
                vis_chroma_safe(a_ref_l),
                os.path.join(img_dir, "a_ref_low.png")
            )
            save_image(
                vis_chroma_safe(a_syn_l),
                os.path.join(img_dir, "a_syn_low.png")
            )
            save_image(
                vis_chroma_safe(a_ref_h),
                os.path.join(img_dir, "a_ref_high.png")
            )

        # ----------------------------------------------------
        # Metrics (UNCHANGED)
        # ----------------------------------------------------
        def _to_u8(t):
            t = (t.clamp(0, 1) * 255).byte().squeeze(0)
            return t.permute(1, 2, 0).cpu().numpy()

        gt_u8 = _to_u8(gt)
        ref_u8 = _to_u8(I_ref)
        syn_u8 = _to_u8(I_syn)

        acc["R_PSNR"].append(compare_psnr(gt_u8, ref_u8, data_range=255))
        acc["R_SSIM"].append(compare_ssim(gt_u8, ref_u8, channel_axis=2, data_range=255))
        acc["S_PSNR"].append(compare_psnr(gt_u8, syn_u8, data_range=255))
        acc["S_SSIM"].append(compare_ssim(syn_u8, gt_u8, channel_axis=2, data_range=255))

        ref_chw = I_ref.squeeze(0).cpu().clamp(0, 1)
        syn_chw = I_syn.squeeze(0).cpu().clamp(0, 1)

        r_uciqe = calculate_uciqe(ref_chw)
        r_uiqm = calculate_uiqm(ref_chw)
        s_uciqe = calculate_uciqe(syn_chw)
        s_uiqm = calculate_uiqm(syn_chw)

        acc["R_UCIQE"].append(r_uciqe)
        acc["R_UIQM"].append(r_uiqm)
        acc["S_UCIQE"].append(s_uciqe)
        acc["S_UIQM"].append(s_uiqm)

        if (r_uciqe > s_uciqe) and (r_uiqm > s_uiqm):
            perceptual_win += 1

        e_ref = structure_energy(ref_chw)
        e_syn = structure_energy(syn_chw)
        gain = e_ref - e_syn
        acc["StructGain"].append(gain)
        if gain > 0:
            struct_gain_win += 1

        if hasattr(model.stage2, "last_attn_entropy"):
            acc["AttnEntropy"].append(model.stage2.last_attn_entropy.item())

        total += 1

        pbar.set_postfix({
            "R_PSNR": f"{np.mean(acc['R_PSNR']):.2f}",
            "S_PSNR": f"{np.mean(acc['S_PSNR']):.2f}",
            "Gain%": f"{struct_gain_win / total:.2f}",
            "H(attn)": f"{np.mean(acc['AttnEntropy']):.3f}" if acc["AttnEntropy"] else "N/A",
        })

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary = {k: float(np.mean(v)) if len(v) else 0.0 for k, v in acc.items()}
    summary["StructGainRatio"] = struct_gain_win / total
    summary["PerceptualWinRatio"] = perceptual_win / total

    summary_path = os.path.join(base_dir, "metrics_fast.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write("epoch," + ",".join(summary.keys()) + "\n")

    with open(summary_path, "a") as f:
        f.write(ep + "," + ",".join(f"{summary[k]:.6f}" for k in summary) + "\n")

    print(
        f"[Fast Eval Done] "
        f"StructGain={summary['StructGain']:.6f}, "
        f"P_gain={summary['StructGainRatio']:.3f}, "
        f"P_UIQM+UCIQE={summary['PerceptualWinRatio']:.3f}"
    )

    return summary
