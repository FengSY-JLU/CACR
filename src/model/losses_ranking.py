import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np


# =========================================================
# ✅ Quality Score Functions
# =========================================================

# def color_score(img):
#     # 灰度世界偏差（越小越好 → 取负）
#     mean = img.mean(dim=[2,3])  # (B,3)
#     gray = mean.mean(dim=1, keepdim=True)
#     return - (mean - gray).abs().mean(dim=1)  # (B,)


def contrast_score(img):
    return img.std(dim=[1,2,3])  # (B,)


def entropy_score(img):
    B = img.shape[0]
    img = img.view(B, -1)

    p = img / (img.sum(dim=1, keepdim=True) + 1e-6)
    ent = - (p * torch.log(p + 1e-6)).sum(dim=1)

    return ent

def chroma_score(img):
    # RGB → Lab（近似）
    R, G, B = img[:,0], img[:,1], img[:,2]

    rg = R - G
    yb = 0.5*(R + G) - B

    return torch.sqrt(rg**2 + yb**2).mean(dim=[1,2])

def color_balance_score(img):
    std = img.std(dim=[2,3])  # (B,3)
    return - std.std(dim=1)   # 通道之间越接近越好

def naturalness_score(img):
    mean = img.mean(dim=[2,3])
    std  = img.std(dim=[2,3])

    return - ((mean - 0.5).abs().mean(dim=1) + (std - 0.2).abs().mean(dim=1))

# =========================================================
# ✅ Offsets Quality Score
# =========================================================

# def compute_quality_scores(I_k_list):
#     scores = []

#     for img in I_k_list:
#         s = (
#             0.5 * color_score(img) +
#             0.3 * contrast_score(img) +
#             0.2 * entropy_score(img)
#         )
#         scores.append(s)

#     scores = torch.stack(scores, dim=1)  # (B,K)
#     scores = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True) + 1e-6)
#     return scores

def compute_quality_scores(I_k_list):
    scores = []

    for img in I_k_list:
        s = (
            0.4 * chroma_score(img) +
            0.2 * contrast_score(img) +
            0.2 * entropy_score(img) +
            0.1 * color_balance_score(img) +
            0.1 * naturalness_score(img)
        )
        scores.append(s)

    scores = torch.stack(scores, dim=1)
    scores = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True) + 1e-6)

    return scores


# =========================================================
# ✅ Direction Quality Score
# =========================================================
# def compute_direction_scores(dir_outputs, I_synthesis):
#     scores = []

#     for img in dir_outputs:
#         # ✅ 结构一致（per-sample）
#         s_struct = - (img - I_synthesis).abs().mean(dim=[1,2,3])

#         # ✅ 偏移幅度
#         s_dev = - (img - I_synthesis).abs().mean(dim=[1,2,3])

#         # ✅ 对比度
#         s_contrast = img.std(dim=[1,2,3])

#         s = 0.5 * s_struct + 0.3 * s_dev + 0.2 * s_contrast
#         scores.append(s)

#     return torch.stack(scores, dim=1)  # (B, D)

def compute_refinement_scores(dir_outputs, I_synthesis):

    scores = []

    for img in dir_outputs:

        # improvement over synthesis
        gain = (img - I_synthesis).abs().mean(dim=[1,2,3])

        # contrast increase
        contrast = img.std(dim=[1,2,3]) - I_synthesis.std(dim=[1,2,3])

        # color expansion
        chroma = chroma_score(img) - chroma_score(I_synthesis)

        s = (
            0.5 * gain +
            0.3 * contrast +
            0.2 * chroma
        )

        scores.append(s)

    return torch.stack(scores, dim=1)

def compute_quality_absolute(img):
    return (
        0.4 * chroma_score(img) +
        0.2 * contrast_score(img) +
        0.2 * entropy_score(img) +
        0.1 * color_balance_score(img) +
        0.1 * naturalness_score(img)
    )

# ============================================================
# Light-weight Quality for contribution computation
# ============================================================
def quality_score_light(I,eps=1e-6):
    """轻量化，用于 compute_offset_contribution"""
    chroma = chroma_score(I)
    contrast = contrast_score(I)
    entropy = entropy_score(I)

    score = 0.5*chroma + 0.3*contrast + 0.2*entropy
    return score

def enhancement_relative_score(
    I_ref,
    I_syn,
):

    # --------------------------------
    # contrast gain
    # --------------------------------
    contrast_gain = (
        contrast_score(I_ref)
        - contrast_score(I_syn)
    )

    # --------------------------------
    # chroma gain
    # --------------------------------
    chroma_gain = (
        chroma_score(I_ref)
        - chroma_score(I_syn)
    )

    # --------------------------------
    # entropy gain
    # --------------------------------
    entropy_gain = (
        entropy_score(I_ref)
        - entropy_score(I_syn)
    )

    # --------------------------------
    # structure preservation
    # prevent hallucination
    # --------------------------------
    structure = -(
        torch.abs(I_ref - I_syn)
    ).mean(dim=[1,2,3])

    score = (
        0.4 * contrast_gain
        + 0.4 * chroma_gain
        + 0.1 * entropy_gain
        + 0.1 * structure
    )

    return score


# ============================================================
# Enhancement-Oriented Relative Quality Score
# Inspired by UIQM / UCIQE / UIEQ
# ============================================================

def enhancement_perceptual_score(
    I_ref,
    I_syn,
    eps=1e-6
):
    """
    Relative enhancement score.

    Goal:
        refined should be perceptually better than synthesis

    Inspired by:
        - UIQM
        - UCIQE
        - UIEQ

    but NOT directly optimizing them.

    Returns:
        score: (B,)
    """

    # ---------------------------------------------------
    # 1. chroma improvement
    # (UICM / UCIQE colorfulness)
    # ---------------------------------------------------
    chroma_gain = (
        chroma_score(I_ref)
        - chroma_score(I_syn)
    )

    # ---------------------------------------------------
    # 2. contrast improvement
    # (UIConM / UCIQE contrast)
    # ---------------------------------------------------
    contrast_gain = (
        contrast_score(I_ref)
        - contrast_score(I_syn)
    )

    # ---------------------------------------------------
    # 3. entropy improvement
    # (information richness)
    # ---------------------------------------------------
    entropy_gain = (
        entropy_score(I_ref)
        - entropy_score(I_syn)
    )

    # ---------------------------------------------------
    # 4. color balance improvement
    # prevent over-saturation
    # ---------------------------------------------------
    balance_gain = (
        color_balance_score(I_ref)
        - color_balance_score(I_syn)
    )

    # ---------------------------------------------------
    # 5. naturalness improvement
    # prevent extreme illumination
    # ---------------------------------------------------
    natural_gain = (
        naturalness_score(I_ref)
        - naturalness_score(I_syn)
    )

    # ---------------------------------------------------
    # 6. structure preservation
    # prevent hallucination
    # ---------------------------------------------------
    structure_preserve = -(
        torch.abs(I_ref - I_syn)
    ).mean(dim=[1,2,3])

    # ---------------------------------------------------
    # final score
    # ---------------------------------------------------
    score = (
        0.30 * chroma_gain
        + 0.30 * contrast_gain
        + 0.15 * entropy_gain
        + 0.10 * balance_gain
        + 0.10 * natural_gain
        + 0.05 * structure_preserve
    )

    return score

# =========================================================
# ✅ Debug Visualization
# =========================================================

def save_debug_images(
    I_input,
    I_synthesis,
    I_refined,
    offsets,
    debug_info,
    save_dir,
    step,
    max_save=2
):
    """
    保存可视化：
    - synthesis
    - refined
    - I_k (每个方向)
    """

    os.makedirs(save_dir, exist_ok=True)

    B = I_input.shape[0]
    K = len(offsets)

    scores = debug_info["scores"].cpu().numpy()
    att    = debug_info["att"].cpu().numpy()

    for b in range(min(B, max_save)):
        base = f"{save_dir}/step{step}_b{b}"

        def to_img(x):
            x = x[b].detach().cpu().permute(1,2,0).numpy()
            x = np.clip(x, 0, 1)
            x = (x * 255).astype(np.uint8)
            return x

        # 保存主图
        cv2.imwrite(base + "_syn.png", to_img(I_synthesis))
        cv2.imwrite(base + "_ref.png", to_img(I_refined))

        # 每个方向
        for k in range(K):
            I_k = I_synthesis + offsets[k]
            img = to_img(I_k)

            cv2.imwrite(
                base + f"_k{k}_score{scores[b,k]:.3f}_att{att[b,k]:.3f}.png",
                img
            )
