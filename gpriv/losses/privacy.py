"""Angular gender-privacy constraints Lp1 / Lp2 (paper Alg. 1–2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _safe_centroid(emb: torch.Tensor) -> torch.Tensor:
    if emb.numel() == 0:
        return emb.new_zeros(emb.shape[-1] if emb.dim() else 0)
    c = emb.mean(dim=0)
    return F.normalize(c, p=2, dim=0)


def lp1_loss(emb: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
    """Angle between male/female batch centroids."""
    male = emb[gender == 1]
    female = emb[gender == 0]
    if male.size(0) == 0 or female.size(0) == 0:
        return emb.new_zeros(())
    cm = _safe_centroid(male)
    cf = _safe_centroid(female)
    return (cm * cf).sum().clamp(-1 + 1e-6, 1 - 1e-6).acos()


def lp2_loss(
    emb: torch.Tensor,
    gender: torch.Tensor,
    weight: torch.Tensor,
    class_gender: torch.Tensor,
) -> torch.Tensor:
    """
    Alg. 2: angle from each sample to the centroid of the opposite gender's identity
    weights, taken over all identities; class_gender gives the gender of each row of weight.
    """
    w = F.normalize(weight, p=2, dim=1)
    male_mask = gender == 1
    female_mask = gender == 0
    if male_mask.sum() == 0 or female_mask.sum() == 0:
        return emb.new_zeros(())

    cm_w = _safe_centroid(w[class_gender == 1])
    cf_w = _safe_centroid(w[class_gender == 0])

    # male samples → female weight centroid; female → male
    loss_m = (emb[male_mask] @ cf_w).clamp(-1 + 1e-6, 1 - 1e-6).acos().mean()
    loss_f = (emb[female_mask] @ cm_w).clamp(-1 + 1e-6, 1 - 1e-6).acos().mean()
    return loss_m + loss_f


def privacy_losses(
    emb: torch.Tensor,
    gender: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    class_gender: torch.Tensor,
    alpha: float,
    beta: float,
) -> tuple[torch.Tensor, dict]:
    lr = F.cross_entropy(logits, labels)
    p1 = lp1_loss(emb, gender)
    p2 = lp2_loss(emb, gender, weight, class_gender)
    total = lr + alpha * p1 + beta * p2
    return total, {
        "lr": float(lr.detach()),
        "lp1": float(p1.detach()),
        "lp2": float(p2.detach()),
        "total": float(total.detach()),
    }
