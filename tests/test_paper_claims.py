"""The paper's central claim, reproduced with this repo's losses on synthetic embeddings.

Abstract: the constraints make gender harder to infer while verification stays high.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gpriv.eval import best_accuracy, gender_accg
from gpriv.losses.privacy import privacy_losses
from gpriv.models.privacy_head import NormSoftmax, PrivacyProjector

DIM, IDS, TRAIN_SAMPLES, TEST_SAMPLES = 64, 40, 10, 10


def make_features(seed: int = 0):
    rng = np.random.RandomState(seed)
    centers = rng.normal(size=(IDS, DIM))
    gender_dir = rng.normal(size=DIM)
    id_gender = np.arange(IDS) % 2
    centers += 1.0 * np.where(id_gender[:, None] == 1, 1.0, -1.0) * gender_dir

    def sample(n: int):
        labels = np.repeat(np.arange(IDS), n)
        x = centers[labels] + 0.5 * rng.normal(size=(len(labels), DIM))
        return x.astype(np.float32), labels, id_gender[labels]

    return sample(TRAIN_SAMPLES), sample(TEST_SAMPLES)


def verification_accuracy(emb: np.ndarray, labels: np.ndarray) -> float:
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    scores = emb @ emb.T
    i, j = np.triu_indices(len(emb), k=1)
    return best_accuracy(scores[i, j], (labels[i] == labels[j]).astype(int))[0]


@pytest.fixture(scope="module")
def outcome():
    torch.manual_seed(0)
    (x, y, g), (x_test, y_test, g_test) = make_features()
    projector = PrivacyProjector(DIM, DIM)
    head = NormSoftmax(DIM, IDS, s=64.0)
    opt = torch.optim.Adam(list(projector.parameters()) + list(head.parameters()), lr=1e-3)
    xs, ys, gs = torch.tensor(x), torch.tensor(y), torch.tensor(g)
    for _ in range(400):
        emb = projector(xs)
        loss, _ = privacy_losses(emb, gs, head(emb), ys, head.weight, alpha=20.0, beta=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        private = projector(torch.tensor(x_test)).numpy()
    return x_test, private, y_test, g_test


def test_constraints_reduce_gender_predictability(outcome):
    original, private, _, gender = outcome
    assert gender_accg(private, gender)["accg"] <= gender_accg(original, gender)["accg"] - 0.10


def test_constraints_keep_verification_accuracy(outcome):
    original, private, labels, _ = outcome
    assert verification_accuracy(private, labels) >= verification_accuracy(original, labels) - 0.02


def test_lp1_pulls_gender_centroids_together(outcome):
    original, private, _, gender = outcome

    def centroid_angle(emb: np.ndarray) -> float:
        e = F.normalize(torch.tensor(emb), dim=1)
        cos = F.cosine_similarity(e[gender == 0].mean(0), e[gender == 1].mean(0), dim=0)
        return float(cos.clamp(-1, 1).acos())

    assert centroid_angle(private) < centroid_angle(original)
