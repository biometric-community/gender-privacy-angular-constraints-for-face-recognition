"""Embed one image with original or private features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gpriv.models.backbone import ResNet18Encoder
from gpriv.models.privacy_head import PrivacyProjector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--image", required=True)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = int(cfg["model"].get("embed_dim", 512))
    private_dim = int(cfg["model"].get("private_dim", 512))
    ckpt = torch.load(
        Path(cfg["paths"]["checkpoint_dir"]) / "last.pt",
        map_location=device,
        weights_only=False,
    )
    backbone = ResNet18Encoder(embed_dim, pretrained=False).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    backbone.eval()
    projector = PrivacyProjector(embed_dim, private_dim).to(device)
    projector.load_state_dict(ckpt["projector"])
    projector.eval()
    tf = transforms.Compose(
        [
            transforms.Resize((int(cfg["model"].get("image_size", 112)),) * 2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        e = backbone(x)
        if args.private:
            e = projector(e)
    out = {"embedding": e.cpu().numpy().tolist()[0], "private": bool(args.private)}
    text = json.dumps(out)
    print(text[:200] + "...")
    if args.out:
        Path(args.out).write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
