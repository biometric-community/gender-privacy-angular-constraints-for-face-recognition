# Gender Privacy: Angular Constraints for Face Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

PyTorch rebuild of angular gender-privacy constraints (Lp1/Lp2) for face embeddings.

**Repository:** https://github.com/biometric-community/gender-privacy-angular-constraints-for-face-recognition

## Setup

```bash
bash scripts/setup_env.sh
```

## Data

- CelebA: `../../datasets/celeba/extracted/celeba`
- LFW: `../../datasets/lfw`

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md).

## Citation

See paper DOI in `papers/gender-privacy-angular-constraints-for-face-recognition/analysis.md`.
