# Config — ResNet-18 + concat

| Setting | Value |
|---|---|
| Predictor class | `SequencePredictor (concat fusion)` |
| Visual encoder | ResNet-18 (frozen, pretrained) |
| Latent dim | 16 |
| GRU hidden dim | 16 |
| Text decoder | frozen |
| Extra loss term | none |
| Epochs | 35 |

Shared across all experiments unless noted above: LATENT_DIM default 16, EMB_DIM 16, BATCH_SIZE 8, LR 0.001, optimizer Adam, criterion_images L1Loss, criterion_text CrossEntropyLoss (pad-masked)
