# 3D Defect Generation — V1

A reproducible starter project for volumetric defect synthesis and 3D deep learning.

## V1 capabilities

- procedural CT-like 3D volume generation
- spherical defects
- ellipsoidal defects
- crack-like defects
- binary 3D defect masks
- lightweight 3D convolutional encoder-decoder
- train / validation / test split
- MAE / RMSE reconstruction metrics
- central-slice visualization

## Pipeline

```text
Clean synthetic volume
        ↓
Procedural defect injection
        ↓
Defective target volume + mask
        ↓
3D Conv Encoder
        ↓
Latent representation
        ↓
3D Decoder
        ↓
Generated defective volume
```

## Run

```bash
pip install -r requirements.txt
python scripts/generate_examples.py
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
```

## Important

This V1 is a technical portfolio baseline, not a claim of physically exact CT defect synthesis.

A stronger V2 should move toward:
- conditional 3D VAE,
- 3D diffusion,
- defect-conditioned generation,
- material-aware synthesis,
- scanner/noise simulation,
- synthetic-to-real validation.
