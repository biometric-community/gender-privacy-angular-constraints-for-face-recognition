"""Stage A: backbone ArcFace on CelebA; Stage B: freeze + privacy projector."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gpriv.data.celeba import CelebAGenderDataset, resolve_celeba_root
from gpriv.losses.privacy import privacy_losses
from gpriv.models.backbone import ResNet18Encoder
from gpriv.models.privacy_head import NormSoftmax, PrivacyProjector


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def train(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    root = resolve_celeba_root(cfg)
    seed = int(cfg["train"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    max_n = cfg["data"].get("max_images")
    max_n = int(max_n) if max_n is not None else None
    image_size = int(cfg["model"].get("image_size", 112))
    embed_dim = int(cfg["model"].get("embed_dim", 512))
    private_dim = int(cfg["model"].get("private_dim", 512))

    train_ds = CelebAGenderDataset(
        root,
        split="train",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_n,
        seed=seed,
        train=True,
    )
    # gender-balanced sampling (paper uses gender-balanced batches)
    genders = np.array([g for _, _, g in train_ds.samples], dtype=np.int64)
    n_m = max(1, int((genders == 1).sum()))
    n_f = max(1, int((genders == 0).sum()))
    w = np.where(genders == 1, 0.5 / n_m, 0.5 / n_f).astype(np.float64)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 128)),
        sampler=sampler,
        num_workers=int(cfg["train"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    backbone = ResNet18Encoder(
        embed_dim=embed_dim,
        pretrained=bool(cfg["model"].get("pretrained_backbone", True)),
    ).to(device)
    # temporary identity head for stage A
    head_a = NormSoftmax(
        embed_dim,
        train_ds.num_classes,
        s=float(cfg["model"].get("arcface_s", 64.0)),
    ).to(device)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    history = []
    t0 = time.time()
    lr_b = float(cfg["train"].get("lr_backbone", 0.1))
    mom = float(cfg["train"].get("momentum", 0.9))
    wd = float(cfg["train"].get("weight_decay", 5e-4))
    opt_a = torch.optim.SGD(
        list(backbone.parameters()) + list(head_a.parameters()),
        lr=lr_b,
        momentum=mom,
        weight_decay=wd,
    )
    epochs_a = int(cfg["train"].get("backbone_epochs", 8))

    backbone.train()
    head_a.train()
    for epoch in range(1, epochs_a + 1):
        losses = []
        pbar = tqdm(loader, desc=f"backbone {epoch}/{epochs_a}", leave=False)
        for images, labels, _gender in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            emb = backbone(images)
            logits = head_a(emb)
            loss = F.cross_entropy(logits, labels)
            opt_a.zero_grad(set_to_none=True)
            loss.backward()
            opt_a.step()
            losses.append(float(loss.detach()))
            pbar.set_postfix(loss=f"{losses[-1]:.3f}")
        mean_l = float(np.mean(losses))
        history.append({"stage": "backbone", "epoch": epoch, "total": mean_l})
        print(f"[backbone] epoch={epoch} loss={mean_l:.4f}")

    # Stage B: freeze backbone, train privacy projector
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    projector = PrivacyProjector(embed_dim, private_dim).to(device)
    head_b = NormSoftmax(
        private_dim,
        train_ds.num_classes,
        s=float(cfg["model"].get("arcface_s", 64.0)),
    ).to(device)
    lr_p = float(cfg["train"].get("lr_privacy", 0.01))
    opt_b = torch.optim.SGD(
        list(projector.parameters()) + list(head_b.parameters()),
        lr=lr_p,
        momentum=mom,
        weight_decay=wd,
    )
    alpha = float(cfg["train"].get("alpha", 20.0))
    beta = float(cfg["train"].get("beta", 1.0))
    epochs_b = int(cfg["train"].get("privacy_epochs", 40))

    projector.train()
    head_b.train()
    for epoch in range(1, epochs_b + 1):
        epoch_parts = []
        pbar = tqdm(loader, desc=f"privacy {epoch}/{epochs_b}", leave=False)
        for images, labels, gender in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            gender = gender.to(device, non_blocking=True)
            with torch.no_grad():
                base = backbone(images)
            priv = projector(base)
            logits = head_b(priv)
            loss, parts = privacy_losses(
                priv, gender, logits, labels, head_b.weight, alpha, beta
            )
            opt_b.zero_grad(set_to_none=True)
            loss.backward()
            opt_b.step()
            epoch_parts.append(parts)
            pbar.set_postfix(loss=f"{parts['total']:.3f}")
        mean = {k: float(np.mean([d[k] for d in epoch_parts])) for k in epoch_parts[0]}
        mean.update({"stage": "privacy", "epoch": epoch})
        history.append(mean)
        print(
            f"[privacy] epoch={epoch} total={mean['total']:.4f} "
            f"lr={mean['lr']:.4f} lp1={mean['lp1']:.4f} lp2={mean['lp2']:.4f}"
        )

    elapsed = time.time() - t0
    ckpt = {
        "backbone": backbone.state_dict(),
        "projector": projector.state_dict(),
        "head": head_b.state_dict(),
        "num_classes": train_ds.num_classes,
        "config": cfg,
    }
    torch.save(ckpt, ckpt_dir / "last.pt")
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "backbone_epochs": epochs_a,
        "privacy_epochs": epochs_b,
        "n_train": len(train_ds),
        "num_classes": train_ds.num_classes,
        "alpha": alpha,
        "beta": beta,
        "elapsed_sec": elapsed,
        "history": history,
        "checkpoint": str(ckpt_dir / "last.pt"),
    }
    with open(log_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {log_dir / 'train_summary.json'} elapsed={elapsed:.1f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    train(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
