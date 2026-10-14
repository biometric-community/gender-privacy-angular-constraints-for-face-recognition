"""Figures + REPORT.md from real metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gpriv.plot_style import (
    FIGSIZE,
    LINEWIDTH_MAIN,
    MULTI_SERIES_COLORS,
    apply_rcparams,
    apply_style,
)


def plot_train(history, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for stage, color in [("backbone", MULTI_SERIES_COLORS[0]), ("privacy", MULTI_SERIES_COLORS[1])]:
        hs = [h for h in history if h.get("stage") == stage]
        if hs:
            ax.plot(
                [h["epoch"] for h in hs],
                [h["total"] for h in hs],
                color=color,
                linewidth=LINEWIDTH_MAIN,
                label=stage,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_tradeoff(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    tags = ["original", "private"]
    accg = [eval_res[t]["gender"]["accg"] * 100 for t in tags]
    eer = [eval_res[t]["lfw_eer"] * 100 for t in tags]
    x = np.arange(2)
    axes[0].bar(x, accg, color=MULTI_SERIES_COLORS[:2], edgecolor="#002080")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Original", "Private"])
    axes[0].set_ylabel("ACCG (%)")
    axes[0].set_title("Gender balanced accuracy (↓ better privacy)")
    apply_style(axes[0], grid_axis="y")
    axes[1].bar(x, eer, color=MULTI_SERIES_COLORS[:2], edgecolor="#002080")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Original", "Private"])
    axes[1].set_ylabel("LFW EER (%)")
    axes[1].set_title("Verification EER (↓ better utility)")
    apply_style(axes[1], grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, report_path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    o, p = eval_res["original"], eval_res["private"]
    pic = eval_res.get("pic", {})
    lines = [
        f"# REPORT: Gender Privacy Angular Constraints — {protocol}",
        "",
        f"**Protocol:** `{protocol}`  ",
        f"**Package:** `gpriv`  ",
        "",
        "## Summary",
        "",
        "| Embedding | LFW Acc | LFW EER | ACCG |",
        "|-----------|---------|---------|------|",
        f"| Original | {o['lfw_accuracy']*100:.2f}% | {o['lfw_eer']*100:.2f}% | {o['gender']['accg']*100:.2f}% |",
        f"| Private (Lp1+Lp2) | {p['lfw_accuracy']*100:.2f}% | {p['lfw_eer']*100:.2f}% | {p['gender']['accg']*100:.2f}% |",
        "",
        f"- PIC proxy (PG−IL): **{pic.get('pic', float('nan')):.4f}** (PG={pic.get('pg', float('nan')):.4f}, IL={pic.get('il', float('nan')):.4f})",
        "",
        "## Training",
        "",
        f"- Backbone epochs: {train_s.get('backbone_epochs')}",
        f"- Privacy epochs: {train_s.get('privacy_epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- alpha / beta: {train_s.get('alpha')} / {train_s.get('beta')}",
        f"- Elapsed: {train_s.get('elapsed_sec', 0):.2f} s",
        f"- Checkpoint: `{train_s.get('checkpoint')}`",
        "",
        "## Figures",
        "",
        "### `train_loss.svg`",
        "",
        "![train_loss](figures/train_loss.svg)",
        "",
        "### `privacy_utility.svg`",
        "",
        "![privacy_utility](figures/privacy_utility.svg)",
        "",
        "## Reproduce",
        "",
        "```bash",
        "bash scripts/smoke.sh",
        "bash scripts/train_full.sh",
        "bash scripts/report.sh configs/default.yaml",
        "```",
        "",
        "## Notes",
        "",
        "- CelebA for train + gender probe; LFW pairs for verification.",
        "- VGGFace2 / AgeDB / ColorFeret omitted (see DEVIATIONS.md).",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval_results.json").read_text())
    plot_train(train_s.get("history", []), fig_dir / "train_loss.svg")
    plot_tradeoff(eval_res, fig_dir / "privacy_utility.svg")
    tracked = _ROOT / "figures"
    tracked.mkdir(exist_ok=True)
    for name in ("train_loss.svg", "privacy_utility.svg"):
        src = fig_dir / name
        if src.exists():
            (tracked / name).write_bytes(src.read_bytes())
    write_report(cfg, train_s, eval_res, _ROOT / "REPORT.md")
    print(f"Wrote {_ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
