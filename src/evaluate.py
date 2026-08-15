from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.models.autoencoder3d import DefectAutoencoder3D


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out = Path(cfg["output"]["directory"])

    clean = np.load(out / "test_clean.npy")
    target = np.load(out / "test_defective.npy")
    masks = np.load(out / "test_masks.npy")

    ckpt = torch.load(out / "autoencoder3d_best.pt", map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DefectAutoencoder3D(
        latent_channels=cfg["model"]["latent_channels"],
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    with torch.no_grad():
        pred = model(torch.as_tensor(clean[:, None], dtype=torch.float32, device=device))
        pred = pred.cpu().numpy()[:, 0]

    metrics = {
        "MAE": float(mean_absolute_error(target.reshape(-1), pred.reshape(-1))),
        "RMSE": float(np.sqrt(mean_squared_error(target.reshape(-1), pred.reshape(-1)))),
    }

    print(json.dumps(metrics, indent=2))

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    np.save(out / "test_generated.npy", pred)

    idx = 0
    z = clean.shape[1] // 2

    for name, volume in [
        ("clean", clean[idx]),
        ("target_defect", target[idx]),
        ("generated", pred[idx]),
        ("mask", masks[idx]),
    ]:
        plt.figure(figsize=(5, 5))
        plt.imshow(volume[z], cmap="gray")
        plt.title(f"{name} - central slice")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out / f"{name}_slice.png", dpi=180)
        plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    main(p.parse_args().config)
