# DEVIATIONS — Gender Privacy Angular Constraints

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Train data | VGGFace2 / AgeDB / ColorFeret | CelebA identities + Male | VGGFace2/AgeDB empty locally |
| D2 | Backbone | ArcFace IResNet50 (pretrained) | ResNet-18 trained on CelebA then frozen | No MS1M/VGGFace2 weights |
| D3 | Gender ACCG | LinearSVM + LogReg + RBF SVM | LinearSVM + LogReg | RBF SVM costly; 3-fold CV retained |
| D4 | Privacy epochs | 100 | 40 (full) | Practical full-train budget |
| D5 | Eval sets | LFW + AgeDB-30 + ColorFeret + VGGFace2 | LFW + CelebA gender probe | Missing datasets |
