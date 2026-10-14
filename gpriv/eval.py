"""LFW verification + CelebA gender ACCG (linear probe) before/after privacy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gpriv.data.celeba import CelebAGenderDataset, resolve_celeba_root
from gpriv.data.lfw import LFWPairsDataset, resolve_lfw_root
from gpriv.models.backbone import ResNet18Encoder
from gpriv.models.privacy_head import PrivacyProjector


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


@torch.no_grad()
def embed_pairs(backbone, projector, loader, device, use_privacy: bool):
    scores, labels = [], []
    for a, b, y in tqdm(loader, desc="lfw", leave=False):
        a = a.to(device)
        b = b.to(device)
        ea = backbone(a)
        eb = backbone(b)
        if use_privacy and projector is not None:
            ea = projector(ea)
            eb = projector(eb)
        # cosine similarity (embeddings already L2)
        s = (ea * eb).sum(dim=1).cpu().numpy()
        scores.append(s)
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def best_accuracy(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Sweep thresholds; return best accuracy and EER approx."""
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc = 0.0
    best_eer = 1.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        acc = (pred == labels).mean()
        best_acc = max(best_acc, float(acc))
        # FPR / FNR
        pos = labels == 1
        neg = ~pos
        if pos.any() and neg.any():
            fpr = (pred[neg] == 1).mean()
            fnr = (pred[pos] == 0).mean()
            best_eer = min(best_eer, float(0.5 * (fpr + fnr)))
    return best_acc, best_eer


@torch.no_grad()
def extract_celeba_embeddings(backbone, projector, loader, device, use_privacy: bool):
    embs, genders = [], []
    for images, _sid, gender in tqdm(loader, desc="celeba-emb", leave=False):
        images = images.to(device)
        e = backbone(images)
        if use_privacy and projector is not None:
            e = projector(e)
        embs.append(e.cpu().numpy())
        genders.append(gender.numpy())
    return np.concatenate(embs), np.concatenate(genders)


def gender_accg(embs: np.ndarray, gender: np.ndarray, seed: int = 42) -> Dict:
    """3-fold balanced ACC averaged over LinearSVM + LogisticRegression."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = {"linear_svm": [], "logreg": []}
    for train_idx, test_idx in skf.split(embs, gender):
        Xtr, Xte = embs[train_idx], embs[test_idx]
        ytr, yte = gender[train_idx], gender[test_idx]
        svm = LinearSVC(max_iter=2000, dual=False)
        svm.fit(Xtr, ytr)
        scores["linear_svm"].append(
            balanced_accuracy_score(yte, svm.predict(Xte))
        )
        lr = LogisticRegression(max_iter=1000)
        lr.fit(Xtr, ytr)
        scores["logreg"].append(balanced_accuracy_score(yte, lr.predict(Xte)))
    out = {k: float(np.mean(v)) for k, v in scores.items()}
    out["accg"] = float(np.mean(list(out.values())))
    return out


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    celeba = resolve_celeba_root(cfg)
    lfw = resolve_lfw_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    image_size = int(cfg["model"].get("image_size", 112))
    embed_dim = int(cfg["model"].get("embed_dim", 512))
    private_dim = int(cfg["model"].get("private_dim", 512))
    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None

    ckpt_path = Path(cfg["paths"]["checkpoint_dir"]) / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    backbone = ResNet18Encoder(embed_dim=embed_dim, pretrained=False).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    backbone.eval()
    projector = PrivacyProjector(embed_dim, private_dim).to(device)
    projector.load_state_dict(ckpt["projector"])
    projector.eval()

    max_pairs = cfg["eval"].get("max_lfw_pairs")
    max_pairs = int(max_pairs) if max_pairs is not None else None
    lfw_ds = LFWPairsDataset(lfw, image_size=image_size, max_pairs=max_pairs)
    lfw_loader = DataLoader(
        lfw_ds,
        batch_size=int(cfg["eval"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
    )

    results = {"n_lfw_pairs": len(lfw_ds)}
    for use_priv, tag in [(False, "original"), (True, "private")]:
        scores, labels = embed_pairs(backbone, projector, lfw_loader, device, use_priv)
        acc, eer = best_accuracy(scores, labels)
        results[tag] = {"lfw_accuracy": acc, "lfw_eer": eer}
        print(f"[{tag}] LFW acc={acc:.4f} eer={eer:.4f}")

    # gender probe on celeba held-out
    probe_max = cfg["eval"].get("gender_probe_max")
    probe_max = int(probe_max) if probe_max is not None else None
    if max_n is not None and probe_max is not None:
        probe_max = min(probe_max, max_n)
    val_ds = CelebAGenderDataset(
        celeba,
        split="val",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=probe_max,
        seed=seed,
        train=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["eval"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
    )
    for use_priv, tag in [(False, "original"), (True, "private")]:
        embs, gender = extract_celeba_embeddings(
            backbone, projector, val_loader, device, use_priv
        )
        g = gender_accg(embs, gender, seed=seed)
        results[tag]["gender"] = g
        results[tag]["n_gender_probe"] = int(len(gender))
        print(f"[{tag}] ACCG={g['accg']:.4f} svm={g['linear_svm']:.4f} lr={g['logreg']:.4f}")

    # PIC-like: privacy gain - identity loss (higher better)
    pg = results["original"]["gender"]["accg"] - results["private"]["gender"]["accg"]
    il = results["private"]["lfw_eer"] - results["original"]["lfw_eer"]
    results["pic"] = {"pg": float(pg), "il": float(il), "pic": float(pg - il)}

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
