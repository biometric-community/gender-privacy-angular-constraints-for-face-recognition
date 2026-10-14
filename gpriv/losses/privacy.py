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
    """Minimize angle between male/female batch centroids → maximize cosine."""
    male = emb[gender == 1]
    female = emb[gender == 0]
    if male.size(0) == 0 or female.size(0) == 0:
        return emb.new_zeros(())
    cm = _safe_centroid(male)
    cf = _safe_centroid(female)
    # maximize cos → minimize (1 - cos)
    cos = (cm * cf).sum().clamp(-1 + 1e-6, 1 - 1e-6)
    return 1.0 - cos


def lp2_loss(
    emb: torch.Tensor,
    gender: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Push embeddings toward opposing-gender identity-weight centroids.
    Approximate Alg. 2: for each sample, use mean of opposite-gender class weights
    present in the batch as the target direction; maximize cosine to that target.
    """
    w = F.normalize(weight, p=2, dim=1)
    male_mask = gender == 1
    female_mask = gender == 0
    if male_mask.sum() == 0 or female_mask.sum() == 0:
        return emb.new_zeros(())

    male_ids = labels[male_mask].unique()
    female_ids = labels[female_mask].unique()
    if male_ids.numel() == 0 or female_ids.numel() == 0:
        return emb.new_zeros(())

    cm_w = _safe_centroid(w[male_ids])
    cf_w = _safe_centroid(w[female_ids])

    # male samples → female weight centroid; female → male
    loss_m = (1.0 - (emb[male_mask] * cf_w).sum(dim=1)).mean() if male_mask.any() else emb.new_zeros(())
    loss_f = (1.0 - (emb[female_mask] * cm_w).sum(dim=1)).mean() if female_mask.any() else emb.new_zeros(())
    return 0.5 * (loss_m + loss_f)


def privacy_losses(
    emb: torch.Tensor,
    gender: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    alpha: float,
    beta: float,
) -> tuple[torch.Tensor, dict]:
    lr = F.cross_entropy(logits, labels)
    p1 = lp1_loss(emb, gender)
    p2 = lp2_loss(emb, gender, weight, labels)
    total = lr + alpha * p1 + beta * p2
    return total, {
        "lr": float(lr.detach()),
        "lp1": float(p1.detach()),
        "lp2": float(p2.detach()),
        "total": float(total.detach()),
    }
