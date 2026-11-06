"""Constraints, projector and metrics, checked against the authors' published code."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from gpriv.data.lfw import _parse_pairs
from gpriv.eval import best_accuracy, gender_accg
from gpriv.losses.privacy import lp1_loss, lp2_loss
from gpriv.models.privacy_head import PrivacyProjector


def authors_p1(emb: torch.Tensor, gender: torch.Tensor) -> torch.Tensor:
    """utils/losses.py P1Loss: the angle between the two gender centroids."""
    female = F.normalize(emb[gender == 0].mean(dim=0), dim=0)
    male = F.normalize(emb[gender == 1].mean(dim=0), dim=0)
    return (female * male).sum().clamp(-1, 1).acos()


def authors_p2(emb, gender, weight, class_gender) -> torch.Tensor:
    """utils/losses.py P2Loss: angles to the opposite gender's weights over all identities."""
    w = F.normalize(weight, dim=1)
    w_female = F.normalize(w[class_gender == 0].mean(dim=0), dim=0)
    w_male = F.normalize(w[class_gender == 1].mean(dim=0), dim=0)
    female_to_male = (emb[gender == 0] @ w_male).clamp(-1, 1).acos().mean()
    male_to_female = (emb[gender == 1] @ w_female).clamp(-1, 1).acos().mean()
    return female_to_male + male_to_female


@pytest.fixture
def batch():
    torch.manual_seed(0)
    class_gender = torch.arange(20) % 2
    weight = torch.randn(20, 32)
    labels = torch.arange(8).repeat(4)  # only 8 of the 20 identities are in the batch
    emb = F.normalize(torch.randn(32, 32), dim=1)
    return emb, class_gender[labels], weight, labels, class_gender


def test_lp1_is_the_centroid_angle(batch):
    emb, gender, *_ = batch
    assert lp1_loss(emb, gender).item() == pytest.approx(authors_p1(emb, gender).item(), abs=1e-4)


def test_lp2_uses_angles_to_all_opposite_gender_identities(batch):
    emb, gender, weight, labels, class_gender = batch
    expected = authors_p2(emb, gender, weight, class_gender).item()
    assert lp2_loss(emb, gender, weight, labels).item() == pytest.approx(expected, abs=1e-4)


def test_projector_has_two_fc_layers():
    layers = [m for m in PrivacyProjector(64, 64).net if isinstance(m, nn.Linear)]
    assert len(layers) == 2


def test_projector_output_layer_has_no_activation():
    """backbones/iresnet.py ProjectionLayer: LeakyReLU only between the FC layers."""
    assert isinstance(list(PrivacyProjector(64, 64).net)[-1], nn.Linear)


@pytest.mark.parametrize("key", ["alpha", "beta", "arcface_s", "arcface_m", "batch_size", "lr_privacy"])
def test_hyperparameters_match_the_authors(default_cfg, targets, key):
    merged = {**default_cfg["train"], **default_cfg["model"]}
    assert merged[key] == targets["authors_config"][key]


def test_optimizer_matches_the_authors(default_cfg, targets):
    assert default_cfg["train"]["optimizer"].lower() == targets["authors_config"]["optimizer"]


def test_capped_lfw_pairs_keep_both_classes(tmp_path):
    """pairs.txt lists 300 mated then 300 non-mated pairs, so a cap must not cut the second."""
    mated = [f"A {i + 1} {i + 2}" for i in range(300)]
    non_mated = [f"A {i + 1} B {i + 1}" for i in range(300)]
    path = tmp_path / "pairs.txt"
    path.write_text("\n".join(["1\t300", *mated, *non_mated]) + "\n")
    labels = {label for _, _, label in _parse_pairs(path, max_pairs=200)}
    assert labels == {0, 1}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_eer_is_the_point_where_fmr_equals_fnmr():
    lo, hi = 0.0, 1.0
    for _ in range(80):
        t = 0.5 * (lo + hi)
        fmr, fnmr = 1.0 - _norm_cdf(t), _norm_cdf((t - 1.0) / 0.1)
        lo, hi = (t, hi) if fmr > fnmr else (lo, t)
    expected = 1.0 - _norm_cdf(t)

    rng = np.random.RandomState(3)
    scores = np.concatenate([rng.normal(1.0, 0.1, 200000), rng.normal(0.0, 1.0, 200000)])
    labels = np.concatenate([np.ones(200000, dtype=int), np.zeros(200000, dtype=int)])
    _, eer = best_accuracy(scores, labels)
    assert eer == pytest.approx(expected, abs=0.01)


def test_gender_accg_separates_signal_from_chance():
    rng = np.random.RandomState(4)
    gender = rng.randint(0, 2, size=400)
    noise = rng.normal(size=(400, 16))
    signal = noise + 3.0 * gender[:, None]
    assert gender_accg(signal, gender)["accg"] >= 0.95
    assert gender_accg(noise, gender)["accg"] <= 0.60
