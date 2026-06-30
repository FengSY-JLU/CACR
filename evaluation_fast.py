# evaluation_fast.py
# ============================================================
# Fast Evaluation Script (Training / Ablation)
# - 现在支持 evaluation_control.py 中的所有 metrics
# - 动态只计算开启的指标（PSNR/SSIM/MSE + 无参考 + LPIPS）
# - 使用 compute_metrics 统一高效计算
# - 进度条已恢复为原地刷新模式（不会滚动）
# ============================================================

import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.filters import sobel
from skimage.color import rgb2gray

# ====================== 控制文件 ======================
from evaluation_control import ENABLE_METRICS

# ====================== 解决 Restormer Prior 大 tensor cuDNN 报错 ======================
# 必须放在任何模型 forward 之前
try:
    torch.backends.cudnn.enabled = False          # 核心：关闭 cuDNN
    torch.backends.cudnn.benchmark = False        # 关闭自动调优
    torch.backends.cudnn.deterministic = True     # 保证可重复性
    print("[INFO] cuDNN disabled for large convolutions in Restormer prior (stability fix)")
except Exception as e:
    print(f"[WARNING] Failed to disable cuDNN: {e}")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# linear -> sRGB
# ------------------------------------------------------------
try:
    from src.utils.color import linear_to_srgb
except ImportError:
    def linear_to_srgb(x):
        x = torch.clamp(x, 0.0, 1.0)
        mask = (x > 0.0031308).float()
        return torch.clamp(
            (1.055 * x.pow(1.0 / 2.4) - 0.055) * mask + 12.92 * x * (1 - mask),
            0.0, 1.0
        )

# ------------------------------------------------------------
# 统一指标计算（支持你所有 metrics）
# ------------------------------------------------------------
from src.measurement.metrics_utils import compute_metrics

# ------------------------------------------------------------
# Structure energy (benchmark-free)
# ------------------------------------------------------------
def structure_energy(img_chw_srgb):
    img = img_chw_srgb.detach().cpu().numpy().transpose(1, 2, 0)
    gray = rgb2gray(img)
    edge = sobel(gray)
    return np.mean(edge ** 2)


@torch.no_grad()
def evaluate(model, opt, dataloader, epoch=None):
    model.eval()
    ep = str(epoch) if epoch is not None else "N/A"
    print(f"\n[Fast Evaluation] Epoch {ep}")

    base_dir = f"{opt.output_folder.rstrip('/')}_{opt.indicator}"
    epoch_dir = os.path.join(base_dir, f"epoch_{ep}")
    os.makedirs(epoch_dir, exist_ok=True)

    # 动态生成 acc keys（只添加你开启的 metrics）
    acc = {
        "AttnEntropy": [],
        "StructGain": [],
    }
    for prefix in ["R_", "S_"]:
        for metric_name, enabled in ENABLE_METRICS.items():
            if enabled:
                acc[f"{prefix}{metric_name}"] = []

    struct_gain_win = 0
    perceptual_win = 0
    total = 0

    # ====================== 进度条（原地刷新模式） ======================
    pbar = tqdm(
        dataloader,
        ncols=140,
        desc=f"Eval | Epoch {ep}",
        leave=False,          # 评估结束后自动消失，不堆积
        dynamic_ncols=True
    )

    for idx, batch in enumerate(pbar):
        x, gt, name = batch[0].to(device), batch[1].to(device), batch[2]
        img_name = name[0]

        outputs = model(x=x)
        I_ref = outputs["I_refined"]
        I_syn = outputs["I_synthesis"]

        # 检查是否有 NaN
        if torch.isnan(I_ref).any():
            print(f"\n[CRITICAL] Model output contains NaN at Epoch {ep}!")

        # Save images (前 3 张)
        if idx < 9999:
            img_dir = os.path.join(epoch_dir, img_name)
            os.makedirs(img_dir, exist_ok=True)
            save_image(linear_to_srgb(I_ref), os.path.join(img_dir, "refined.png"))
            save_image(linear_to_srgb(I_syn), os.path.join(img_dir, "synthesis.png"))
            for i, p in enumerate(outputs["priors"]):
                save_image(linear_to_srgb(p.clamp(0, 1)), os.path.join(img_dir, f"prior_{i+1}.png"))
            for i, o in enumerate(outputs["offsets_stage2"]):
                save_image(linear_to_srgb(o.clamp(0, 1)), os.path.join(img_dir, f"offset_{i+1}.png"))

        # sRGB 转换
        gt_srgb = linear_to_srgb(gt)
        ref_srgb = linear_to_srgb(I_ref)
        syn_srgb = linear_to_srgb(I_syn)

        def _to_u8(t):
            t = (t.clamp(0, 1) * 255).byte().squeeze(0)
            return t.permute(1, 2, 0).cpu().numpy()

        gt_u8 = _to_u8(gt_srgb)
        ref_u8 = _to_u8(ref_srgb)
        syn_u8 = _to_u8(syn_srgb)

        # ====================== GT 依赖指标 ======================
        if ENABLE_METRICS.get("PSNR", False):
            acc["R_PSNR"].append(compare_psnr(gt_u8, ref_u8, data_range=255))
            acc["S_PSNR"].append(compare_psnr(gt_u8, syn_u8, data_range=255))
        if ENABLE_METRICS.get("SSIM", False):
            acc["R_SSIM"].append(compare_ssim(gt_u8, ref_u8, channel_axis=2, data_range=255))
            acc["S_SSIM"].append(compare_ssim(gt_u8, syn_u8, channel_axis=2, data_range=255))
        if ENABLE_METRICS.get("MSE", False):
            r_mse = np.mean((gt_u8.astype(np.float32) - ref_u8.astype(np.float32)) ** 2)
            s_mse = np.mean((gt_u8.astype(np.float32) - syn_u8.astype(np.float32)) ** 2)
            acc["R_MSE"].append(r_mse)
            acc["S_MSE"].append(s_mse)

        # ====================== 统一计算所有无参考 + LPIPS ======================
        ref_metrics = compute_metrics(
            pred=ref_srgb,
            gt=gt_srgb if ENABLE_METRICS.get("LPIPS", False) else None,
            device=device,
            enable_metrics=ENABLE_METRICS
        )
        syn_metrics = compute_metrics(
            pred=syn_srgb,
            gt=gt_srgb if ENABLE_METRICS.get("LPIPS", False) else None,
            device=device,
            enable_metrics=ENABLE_METRICS
        )

        for m, val in ref_metrics.items():
            acc[f"R_{m}"].append(val)
        for m, val in syn_metrics.items():
            acc[f"S_{m}"].append(val)

        # PerceptualWinRatio
        if ENABLE_METRICS.get("UCIQE", False) and ENABLE_METRICS.get("UIQM", False):
            if (ref_metrics["UCIQE"] > syn_metrics["UCIQE"]) and (ref_metrics["UIQM"] > syn_metrics["UIQM"]):
                perceptual_win += 1

        # ====================== Structure Gain ======================
        ref_chw = ref_srgb.squeeze(0).cpu()
        syn_chw = syn_srgb.squeeze(0).cpu()
        e_ref = structure_energy(ref_chw)
        e_syn = structure_energy(syn_chw)
        gain = e_ref - e_syn
        acc["StructGain"].append(gain)
        if gain > 0:
            struct_gain_win += 1

        if hasattr(model.stage2, "last_attn_entropy"):
            acc["AttnEntropy"].append(model.stage2.last_attn_entropy.item())

        # ====================== CAS ======================
        if hasattr(model.stage2, "last_cas_weights") and model.stage2.last_cas_weights is not None:
            cw = model.stage2.last_cas_weights.detach().cpu()
            if cw.dim() == 4:
                cw = cw.mean(dim=0)
            cas_scalar = cw.view(cw.shape[0], -1).mean(dim=1)

            if "CAS_Entropy" not in acc:
                for i in range(cas_scalar.shape[0]):
                    acc[f"CAS_{i}"] = []
                acc["CAS_Entropy"] = []

            p = cas_scalar / (cas_scalar.sum() + 1e-8)
            H = float(-(p * torch.log(p + 1e-8)).sum())

            for i in range(cas_scalar.shape[0]):
                acc[f"CAS_{i}"].append(cas_scalar[i].item())
            acc["CAS_Entropy"].append(H)

            if idx < 3:
                cas_vis = cas_scalar.view(1, 1, -1)
                save_image(cas_vis, os.path.join(img_dir, "cas_weights.png"), normalize=True)

        total += 1

        # ====================== 实时刷新进度条 ======================
        pbar.set_postfix({
            "R_PSNR": f"{np.mean(acc['R_PSNR']):.2f}" if acc.get("R_PSNR") else "N/A",
            "S_PSNR": f"{np.mean(acc['S_PSNR']):.2f}" if acc.get("S_PSNR") else "N/A",
            "Gain%": f"{struct_gain_win / total:.3f}",
            "P_win": f"{perceptual_win / total:.3f}" if ENABLE_METRICS.get("UCIQE") and ENABLE_METRICS.get("UIQM") else "N/A",
        })

    # ====================== Summary ======================
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