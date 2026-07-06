# src/model/losses.py
# ============================================================
# Final Loss Design for Two-Stage Underwater Color Correction
#
# - B-1: Luminance Anchor (DEFAULT, enabled)
# - B-2: Structure Lower-Bound (OPTIONAL, disabled by default)
# - Refined-only Structure Gain (benchmark-free)
# - Synthesis Structure Floor (very weak, safe)
# - Stage-II Color Freedom (Chroma Expansion)
# - Stage-II Soft Red–Cyan Balance
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
# from src.model.losses_ranking import compute_quality_absolute


# ============================================================
# Color channel helpers
# ============================================================
def red_offset(O):
    """
    O: (B, 3, H, W) color offset
    Positive => red increase relative to cyan
    """
    R = O[:, 0:1]
    G = O[:, 1:2]
    B = O[:, 2:3]
    return R - 0.5 * (G + B)


def cyan_offset(O):
    """
    O: (B, 3, H, W) color offset
    Positive => cyan increase relative to red
    """
    R = O[:, 0:1]
    G = O[:, 1:2]
    B = O[:, 2:3]
    return 0.5 * (G + B) - R


# ============================================================
# CAS weights spatial pooling helper
# ============================================================
def cas_spatial_pool(cas_weights):
    """
    cas_weights: (B, N, H, W)
    return: (B, N)
    """
    return cas_weights.mean(dim=[2, 3])

# ============================================================
# CAS entropy floor loss
# ============================================================
def cas_entropy_floor_loss(
    cas_weights,
    H_target=0.6,
    eps=1e-8,
    reduce="mean"
):
    """
    Prevent CAS from collapsing too early.
    
    cas_weights: Tensor [B, K] or [K]
    """

    if cas_weights.dim() == 1:
        cas_weights = cas_weights.unsqueeze(0)  # [1, K]

    # entropy per sample
    entropy = -torch.sum(
        cas_weights * torch.log(cas_weights + eps),
        dim=1
    )  # [B]

    loss = F.relu(H_target - entropy)

    if reduce == "mean":
        return loss.mean()
    elif reduce == "sum":
        return loss.sum()
    else:
        return loss

# ============================================================
# Color space helpers (Linear RGB -> CIE Lab)
# ============================================================

def linear_rgb_to_xyz(img_lin):
    img = torch.clamp(img_lin, 0.0, 1.0)
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return torch.cat([x, y, z], dim=1)


def xyz_to_lab(xyz):
    xyz_ref = torch.tensor(
        [0.95047, 1.00000, 1.08883],
        device=xyz.device
    ).view(1, 3, 1, 1)

    xyz = xyz / (xyz_ref + 1e-12)
    eps = 0.008856
    kappa = 903.3

    mask = (xyz > eps).float()
    f = xyz.pow(1.0 / 3.0) * mask + ((kappa * xyz + 16) / 116) * (1 - mask)

    L = 116 * f[:, 1:2] - 16
    a = 500 * (f[:, 0:1] - f[:, 1:2])
    b = 200 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([L, a, b], dim=1)


def rgb_to_lab(img):
    return xyz_to_lab(linear_rgb_to_xyz(img))


# ============================================================
# Chroma mask helper (NEW, minimal)
# ============================================================

def chroma_mask(I, tau=3.0):
    """
    Mask out low-chroma (near gray / white) regions.
    These regions should NOT participate in color direction learning.
    """
    lab = rgb_to_lab(I)
    a, b = lab[:, 1:2], lab[:, 2:3]
    C = torch.sqrt(a ** 2 + b ** 2 + 1e-12)
    return (C > tau).float()



def patchwise_gray_world_loss(I, patch=32):
    B, C, H, W = I.shape
    loss, n = 0.0, 0
    for i in range(0, H, patch):
        for j in range(0, W, patch):
            p = I[:, :, i:i + patch, j:j + patch]
            if p.shape[-2] < patch or p.shape[-1] < patch:
                continue
            mu = p.mean(dim=[2, 3])
            loss += F.l1_loss(
                mu,
                mu.mean(dim=1, keepdim=True).expand_as(mu)
            )
            n += 1
    return loss / max(n, 1)


# ============================================================
# Stability anchors
# ============================================================

def luminance_anchor_loss(I_ref, I_syn):
    """
    B-1: SAFE default anchor.
    Prevents synthesis & refined from collapsing together.
    """
    y_ref = 0.299 * I_ref[:, 0] + 0.587 * I_ref[:, 1] + 0.114 * I_ref[:, 2]
    y_syn = 0.299 * I_syn[:, 0] + 0.587 * I_syn[:, 1] + 0.114 * I_syn[:, 2]

    return (
        torch.abs(y_ref.mean() - y_syn.mean()) +
        torch.abs(y_ref.std() - y_syn.std())
    )


def structure_lower_bound_loss(I_ref, I_syn):
    """
    B-2 (OPTIONAL):
    Ensure refined does NOT become structurally worse than synthesis.
    """
    def energy(I):
        y = 0.299 * I[:, 0:1] + 0.587 * I[:, 1:2] + 0.114 * I[:, 2:3]
        gx = torch.abs(y[:, :, 1:] - y[:, :, :-1])
        gy = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
        return gx.mean() + gy.mean()

    return F.relu(energy(I_syn) - energy(I_ref))


# ============================================================
# Refined-only Structure Gain (benchmark-free)
# ============================================================

class RefinedStructureGainLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [[-1,0,1],[-2,0,2],[-1,0,1]],
            dtype=torch.float32
        ).view(1,1,3,3)

        sobel_y = torch.tensor(
            [[-1,-2,-1],[0,0,0],[1,2,1]],
            dtype=torch.float32
        ).view(1,1,3,3)

        self.register_buffer("sx", sobel_x)
        self.register_buffer("sy", sobel_y)

    def energy(self, I):
        y = 0.299*I[:,0:1] + 0.587*I[:,1:2] + 0.114*I[:,2:3]
        gx = F.conv2d(y, self.sx.to(y.device), padding=1)
        gy = F.conv2d(y, self.sy.to(y.device), padding=1)
        return (gx.abs() + gy.abs()).mean(dim=[1,2,3])

    def forward(self, I_ref, I_syn):
        return F.relu(self.energy(I_syn) - self.energy(I_ref)).mean()


_STRUCTURE_GAIN = RefinedStructureGainLoss()


# ============================================================
# NEW: Low-Frequency Color Structure Loss
# ============================================================
def low_freq_color_structure_loss(I_ref, I_syn, k=15, eps=1e-6):
    """
    Transfer low-frequency CHROMA STRUCTURE (direction only).
    Complementary to color direction distillation.
    """
    ref_l = F.avg_pool2d(I_ref, k, stride=1, padding=k//2)
    syn_l = F.avg_pool2d(I_syn, k, stride=1, padding=k//2)

    ref_chroma = ref_l - ref_l.mean(dim=1, keepdim=True)
    syn_chroma = syn_l - syn_l.mean(dim=1, keepdim=True)

    norm_r = torch.norm(ref_chroma, dim=1, keepdim=True) + eps
    norm_s = torch.norm(syn_chroma, dim=1, keepdim=True) + eps

    ref_dir = ref_chroma / norm_r
    syn_dir = syn_chroma / norm_s

    # === NEW: chroma mask ===
    mask = chroma_mask(I_syn.detach())

    return (mask * torch.abs(ref_dir - syn_dir)).mean()


# ============================================================
# NEW: Color Direction Distillation Loss 
# ============================================================
def color_direction_distillation_loss(I_ref, I_syn, eps=1e-8):
    I_ref = torch.clamp(I_ref, 0.0, 1.0)
    I_syn = torch.clamp(I_syn, 0.0, 1.0)

    lab_ref = rgb_to_lab(I_ref)
    lab_syn = rgb_to_lab(I_syn.detach())

    a_r, b_r = lab_ref[:, 1:2], lab_ref[:, 2:3]
    a_s, b_s = lab_syn[:, 1:2], lab_syn[:, 2:3]

    norm_r = torch.sqrt(a_r**2 + b_r**2 + eps)
    norm_s = torch.sqrt(a_s**2 + b_s**2 + eps)
    norm_r = torch.clamp(norm_r, min=1e-6)
    norm_s = torch.clamp(norm_s, min=1e-6)

    dir_r = torch.cat([a_r / norm_r, b_r / norm_r], dim=1)
    dir_s = torch.cat([a_s / norm_s, b_s / norm_s], dim=1)

    mask = chroma_mask(I_syn.detach())
    loss = (mask * torch.abs(dir_r - dir_s)).mean()

    #  nan_to_num
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss


# ============================================================
# Illumination regularization
# ============================================================

def illumination_smoothness_loss(I, k=15):
    L = I.mean(dim=1, keepdim=True)
    L = F.avg_pool2d(L, k, stride=1, padding=k // 2)
    dx = torch.abs(L[:, :, 1:] - L[:, :, :-1])
    dy = torch.abs(L[:, :, :, 1:] - L[:, :, :, :-1])
    return dx.mean() + dy.mean()

def edge_aware_illumination_loss(I, eps=1e-3):
    """
    Edge-aware illumination smoothness.

    Smooth flat regions,
    preserve strong edges.
    """

    # ---------------------------------
    # illumination
    # ---------------------------------
    L = I.mean(dim=1, keepdim=True)

    # low-frequency illumination
    L_blur = F.avg_pool2d(
        L,
        kernel_size=15,
        stride=1,
        padding=7
    )

    # ---------------------------------
    # gradients of illumination
    # ---------------------------------
    dx = L_blur[:, :, :, 1:] - L_blur[:, :, :, :-1]
    dy = L_blur[:, :, 1:, :] - L_blur[:, :, :-1, :]

    # ---------------------------------
    # image edge guidance
    # use original image gradients
    # ---------------------------------
    gray = (
        0.299 * I[:,0:1]
        + 0.587 * I[:,1:2]
        + 0.114 * I[:,2:3]
    )

    gx = torch.abs(
        gray[:, :, :, 1:] - gray[:, :, :, :-1]
    )

    gy = torch.abs(
        gray[:, :, 1:, :] - gray[:, :, :-1, :]
    )

    # ---------------------------------
    # edge-aware weights
    # strong edge -> smaller smoothing
    # ---------------------------------
    wx = torch.exp(-10.0 * gx)
    wy = torch.exp(-10.0 * gy)

    # ---------------------------------
    # smoothness
    # ---------------------------------
    loss_x = (wx * torch.abs(dx)).mean()
    loss_y = (wy * torch.abs(dy)).mean()

    return loss_x + loss_y


def detail_preserve_loss(
    I_ref,
    I_syn,
    threshold=0.03
):
    """
    Preserve strong structural details from synthesis.

    ONLY constrain:
    - strong edges
    - high-frequency structures

    DO NOT force full reconstruction.
    """

    # -----------------------------------
    # luminance
    # -----------------------------------
    Y_ref = (
        0.299 * I_ref[:,0:1] +
        0.587 * I_ref[:,1:2] +
        0.114 * I_ref[:,2:3]
    )

    Y_syn = (
        0.299 * I_syn[:,0:1] +
        0.587 * I_syn[:,1:2] +
        0.114 * I_syn[:,2:3]
    )

    # -----------------------------------
    # gradients
    # -----------------------------------
    gx_syn = torch.abs(Y_syn[:, :, :, 1:] - Y_syn[:, :, :, :-1])
    gy_syn = torch.abs(Y_syn[:, :, 1:, :] - Y_syn[:, :, :-1, :])

    gx_ref = torch.abs(Y_ref[:, :, :, 1:] - Y_ref[:, :, :, :-1])
    gy_ref = torch.abs(Y_ref[:, :, 1:, :] - Y_ref[:, :, :-1, :])

    # -----------------------------------
    # strong-detail mask
    # ONLY preserve strong structures
    # -----------------------------------
    mask_x = (gx_syn > threshold).float()
    mask_y = (gy_syn > threshold).float()

    # -----------------------------------
    # preserve detail magnitude
    # -----------------------------------
    loss_x = (
        mask_x *
        torch.abs(gx_ref - gx_syn.detach())
    ).mean()

    loss_y = (
        mask_y *
        torch.abs(gy_ref - gy_syn.detach())
    ).mean()

    return loss_x + loss_y


def gradient_direction_consistency_loss(
    I_ref,
    I_syn,
    threshold=0.08,
    eps=1e-6
):
    """
    Preserve edge orientation
    WITHOUT forcing full reconstruction.
    """

    # -----------------------------------
    # luminance
    # -----------------------------------
    Y_ref = (
        0.299 * I_ref[:,0:1] +
        0.587 * I_ref[:,1:2] +
        0.114 * I_ref[:,2:3]
    )

    Y_syn = (
        0.299 * I_syn[:,0:1] +
        0.587 * I_syn[:,1:2] +
        0.114 * I_syn[:,2:3]
    )

    # -----------------------------------
    # gradients
    # -----------------------------------
    gx_ref = Y_ref[:,:,:,1:] - Y_ref[:,:,:,:-1]
    gy_ref = Y_ref[:,:,1:,:] - Y_ref[:,:,:-1,:]

    gx_syn = Y_syn[:,:,:,1:] - Y_syn[:,:,:,:-1]
    gy_syn = Y_syn[:,:,1:,:] - Y_syn[:,:,:-1,:]

    # -----------------------------------
    # edge mask
    # -----------------------------------
    mag_syn_x = torch.abs(gx_syn)
    mag_syn_y = torch.abs(gy_syn)

    mask_x = (mag_syn_x > threshold).float()
    mask_y = (mag_syn_y > threshold).float()

    # -----------------------------------
    # normalize direction
    # -----------------------------------
    dir_ref_x = gx_ref / (torch.abs(gx_ref) + eps)
    dir_syn_x = gx_syn / (torch.abs(gx_syn) + eps)

    dir_ref_y = gy_ref / (torch.abs(gy_ref) + eps)
    dir_syn_y = gy_syn / (torch.abs(gy_syn) + eps)

    # -----------------------------------
    # direction consistency
    # -----------------------------------
    loss_x = (
        mask_x *
        torch.abs(dir_ref_x - dir_syn_x.detach())
    ).mean()

    loss_y = (
        mask_y *
        torch.abs(dir_ref_y - dir_syn_y.detach())
    ).mean()

    return loss_x + loss_y


# ============================================================
# relative_improvement_loss
# ============================================================
def relative_improvement_loss(
    I_ref,
    I_syn,
    margin=0.02
):
    """
    Encourage refined image to be
    perceptually better than synthesis.
    """

    from src.model import losses_ranking

    s_ref = losses_ranking.enhancement_perceptual_score(
        I_ref,
        I_syn.detach()
    )

    # synthesis baseline = 0
    # because score is already relative

    loss = F.relu(
        margin - s_ref
    )

    return loss.mean()


# ============================================================
# prior_loss
# ============================================================
def loss_prior(I_refined, prior_list, I_synthesis, temperature=0.1, eps=1e-6):
    """
    Keep refined inside convex hull of priors + synthesis.

    Args:
        I_refined: (B,3,H,W)
        prior_list: list of priors
        I_synthesis: synthesis image
    """

    priors = prior_list + [I_synthesis]

    B = I_refined.shape[0]

    # ----------------------------------
    # flatten
    # ----------------------------------
    ref_flat = I_refined.view(B, -1)

    sims = []

    for P in priors:
        P_flat = P.view(B, -1)

        sim = F.cosine_similarity(ref_flat, P_flat, dim=1)
        sims.append(sim)

    sims = torch.stack(sims, dim=1)  # (B,K)

    # ----------------------------------
    # softmax weights (convex)
    # ----------------------------------
    weights = F.softmax(sims / temperature, dim=1)

    # ----------------------------------
    # reconstruction
    # ----------------------------------
    recon = 0.0
    for k, P in enumerate(priors):
        w = weights[:, k].view(B, 1, 1, 1)
        recon = recon + w * P

    # ----------------------------------
    # loss
    # ----------------------------------
    loss = F.l1_loss(I_refined, recon.detach())

    return loss


# =============================================================
# offsets alignment contribution
# =============================================================

def compute_offset_contribution(
    I_ref,
    weighted_offsets,
    synthesis,
    model,
    quality_fn=None
):
    """
    Estimate contribution of EACH refinement offset.

    IMPORTANT:
    - synthesis is ALWAYS preserved
    - topology is ALWAYS fixed
    - only refinement offsets are ablated

    Args:
        I_ref: (B,3,H,W)
        weighted_offsets: list[(B,3,H,W)]
        synthesis: (B,3,H,W)
        model: full model
        quality_fn: callable

    Returns:
        scores: (B, K)
    """

    if quality_fn is None:
        from src.model import losses_ranking
        quality_fn = losses_ranking.quality_score_light

    scores = []

    with torch.no_grad():

        q_full = quality_fn(I_ref)   # (B,)

        K = len(weighted_offsets)

        for i in range(K):

            # -----------------------------------
            # keep topology fixed
            # -----------------------------------
            subset = []

            for j, off in enumerate(weighted_offsets):

                if i == j:
                    subset.append(torch.zeros_like(off))
                else:
                    subset.append(off)

            # synthesis ALWAYS preserved
            subset.append(synthesis)

            # forward
            out = model.stage2.forward_with_offsets(
                subset
            )["I_refined"]

            q_subset = quality_fn(out)

            # positive => this offset helps
            delta = q_full - q_subset

            scores.append(delta)

    return torch.stack(scores, dim=1)  # (B,K)


def contributor_manifold_loss(
    I_refined,
    prior_list,
    I_synthesis,
    temperature=0.05,
):
    """
    Feasible-solution manifold regularization.

    Goal:
        - refined should stay close to at least ONE
          trusted hypothesis
        - avoid hallucinated colors / unstable solutions
        - DO NOT force convex averaging
        - DO NOT suppress enhancement freedom

    Hypothesis space:
        priors + synthesis

    IMPORTANT:
        This is NOT residual reconstruction.
        This is NOT weighted averaging.
        This is nearest-feasible-solution regularization.
    """

    # --------------------------------------------------
    # trusted hypothesis manifold
    # --------------------------------------------------
    hypothesis_list = prior_list + [I_synthesis]

    dist_list = []

    # --------------------------------------------------
    # compute distance to each trusted solution
    # --------------------------------------------------
    for H in hypothesis_list:

        dist = torch.abs(
            I_refined - H.detach()
        ).mean(dim=[1,2,3])   # (B,)

        dist_list.append(dist)

    dist_stack = torch.stack(dist_list, dim=1)  # (B,K)

    # --------------------------------------------------
    # soft-min
    # allows enhancement freedom
    # while preventing solution-space collapse
    # --------------------------------------------------
    weights = F.softmax(
        -dist_stack / temperature,
        dim=1
    )

    soft_min_dist = (
        weights * dist_stack
    ).sum(dim=1)

    return soft_min_dist.mean()

# ============================================================
# Warm-up helper (NEW, minimal)
# ============================================================

def color_warmup_factor(epoch, start=1, end=9999):
    if epoch < start:
        return 0.0
    if epoch >= end:
        return 1.0
    return float(epoch - start) / float(end - start)

#===========================================================
# Warm-down helper (NEW, minimal)
# ============================================================
def stop_color_warmup(epoch, start=1, end=80):
    if epoch < start:
        return 1.0
    if epoch >= end:
        return 0.0
    return 1.0 - float(epoch - start) / float(end - start)

# ============================================================
# Total loss
# ============================================================

def compute_total_loss(
    I_input,
    I_refined,
    I_synthesis,
    priors,
    weighted_offsets,
    cas_weights,
    attn_offset_only,
    J_phys,
    model=None,
    epoch=1,
    iteration=1,
):
    loss_dict = {}

    # --- Stability anchors ---
    loss_lum = luminance_anchor_loss(I_refined, I_synthesis.detach())

    # --- Refined objective ---
    loss_gain = _STRUCTURE_GAIN(I_refined, I_synthesis.detach())

    # --- CAS spatial pooling (CRITICAL) ---
    if cas_weights is not None:
        prior_weights = cas_spatial_pool(cas_weights)  # (B,N)
    else:
        prior_weights = None

    loss_cas_entropy = cas_entropy_floor_loss(cas_weights, H_target=0.6)

    # --- Color distillation (NEW, CORE) ---
    loss_color_direction = color_direction_distillation_loss(I_refined, I_synthesis)

    # --- Low-frequency color structure (weak) ---
    loss_color_struct = low_freq_color_structure_loss(I_refined, I_synthesis.detach())

    # --- Regularization ---
    loss_illum = edge_aware_illumination_loss(I_refined)

    # =========================================================
    # Hybrid structural anchors
    # synthesis -> enhancement structure
    # input     -> geometry grounding
    # =========================================================

    loss_detail_pre = detail_preserve_loss(
        I_refined,
        I_synthesis
    )

    loss_gra = gradient_direction_consistency_loss(I_refined, I_synthesis)


    loss_rel_improve = relative_improvement_loss(I_refined, I_synthesis,margin=0.05)

    loss_prior_reg = contributor_manifold_loss(I_refined, priors, I_synthesis)


    w = stop_color_warmup(epoch)

    total = (

        # ---------- Stage1 ----------
        0.01 * loss_cas_entropy
        # ---------- Stage2 ----------
        + 0.2 * loss_gain
        + 0.3 * loss_lum
        + 0.0005 * loss_illum
        + 0.15 * loss_detail_pre
        + 0.1 * loss_gra
        + w * 0.005 * loss_prior_reg

        + 0.1 * loss_rel_improve
        # ---------- direction stability ----------
        + 0.05 * loss_color_struct
        + w * 0.005 * loss_color_direction      
    )

    loss_dict = {
        "total_loss": total.item(),
        "lum_anchor": loss_lum.item(),
        "struct_gain": loss_gain.item(),
        "cas_entropy": loss_cas_entropy.item(),
        "rel_improve": loss_rel_improve.item(),
        "color_struct": loss_color_struct.item(),
        "prior_reg": loss_prior_reg.item(),
        "illum_smooth": loss_illum.item(),
        "detail_preseve": loss_detail_pre.item(),
        "gradiant": loss_gra.item(),
        "color_direction": loss_color_direction.item(),
    }

    return total, loss_dict

def safe_loss(loss, name="", eps=1e-6):
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"[LOSS NaN/INF] {name}")
        return torch.zeros_like(loss)
    if loss.abs() > 1e4:
        print(f"[LOSS EXPLODE] {name}: {loss.item():.4e}")
        return torch.clamp(loss, -1e4, 1e4)
    return loss
