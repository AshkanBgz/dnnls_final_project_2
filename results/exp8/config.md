# Config — Perceptual loss (VGG)

| Setting | Value |
|---|---|
| Predictor class | `CrossModalSequencePredictor` |
| Visual encoder | CNN (dual-pathway backbone) |
| Latent dim | 16 |
| GRU hidden dim | 64 |
| Text decoder | unfrozen |
| Extra loss term | VGG perceptual, lambda=0.5 |
| Epochs | 25 |

Shared across all experiments unless noted above: LATENT_DIM default 16, EMB_DIM 16, BATCH_SIZE 8, LR 0.001, optimizer Adam, criterion_images L1Loss, criterion_text CrossEntropyLoss (pad-masked)
