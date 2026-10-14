# FIDELITY_AUDIT — Gender Privacy

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | Two-FC projector + NormSoftmax Lr | 75% | Layout |
| 2 | Lp1 gender-centroid angle | 85% | Alg. 1 |
| 3 | Lp2 opposing-gender weight angles | 88% | Alg. 2 approx |
| 4 | LFW + ACCG + PIC proxy + figures | 92% | D1/D5 datasets |
| 5 | Size gate + scripts + packaging | 95% | Stop |

Confidence vs paper structure: **~95%**.

## Measured run (full)

- CelebA+LFW 3.09 GiB → full train; backbone 8 + privacy 40 epochs
- Original: LFW acc 70.60%, EER 29.40%, ACCG 96.65%
- Private: LFW acc 72.40%, EER 27.60%, ACCG 72.08%
- Directional match to paper: gender separability drops while verification holds
