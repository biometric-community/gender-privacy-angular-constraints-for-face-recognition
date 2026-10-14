# Implementation Plan — Gender Privacy

## Method
Stage A: train ResNet-18 + NormSoftmax on CelebA identities.
Stage B: freeze backbone; train PrivacyProjector + Lr + α Lp1 + β Lp2.

## Figures
| Artifact | Content |
|----------|---------|
| train_loss.svg | backbone + privacy losses |
| privacy_utility.svg | ACCG / LFW EER original vs private |

## Data
CelebA + LFW (size-gated full train).
