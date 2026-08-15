from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import VolumePairDataset
from src.models.autoencoder3d import DefectAutoencoder3D
from src.synthetic import generate_sample


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_data(cfg):
    dc = cfg["data"]
    fc = cfg["defects"]

    clean, defective, masks, labels = [], [], [], []

    for _ in range(int(dc["num_samples"])):
        c, d, m, label = generate_sample(
            size=int(dc["volume_size"]),
            defect_types=list(fc["types"]),
            min_radius=int(fc["min_radius"]),
            max_radius=int(fc["max_radius"]),
            intensity_min=float(fc["intensity_min"]),
            intensity_max=float(fc["intensity_max"]),
            defect_probability=float(dc["defect_probability"]),
        )
        clean.append(c)
        defective.append(d)
        masks.append(m)
        labels.append(label)

    clean = np.stack(clean)
    defective = np.stack(defective)
    masks = np.stack(masks)

    n = len(clean)
    n_train = int(n * dc["train_fraction"])
    n_val = int(n * dc["validation_fraction"])

    return (
        (clean[:n_train], defective[:n_train], masks[:n_train]),
        (clean[n_train:n_train+n_val], defective[n_train:n_train+n_val], masks[n_train:n_train+n_val]),
        (clean[n_train+n_val:], defective[n_train+n_val:], masks[n_train+n_val:]),
        labels,
    )


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    train_pack, val_pack, test_pack, labels = build_data(cfg)

    tr = VolumePairDataset(*train_pack)
    va = VolumePairDataset(*val_pack)
    te = VolumePairDataset(*test_pack)

    tr_loader = DataLoader(tr, batch_size=cfg["training"]["batch_size"], shuffle=True)
    va_loader = DataLoader(va, batch_size=cfg["training"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DefectAutoencoder3D(
        latent_channels=cfg["model"]["latent_channels"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    loss_fn = nn.L1Loss()

    out = Path(cfg["output"]["directory"])
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "autoencoder3d_best.pt"

    best, stale, history = float("inf"), 0, []

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        tr_losses = []

        for clean, defective, _ in tr_loader:
            clean, defective = clean.to(device), defective.to(device)
            optimizer.zero_grad()
            pred = model(clean)
            loss = loss_fn(pred, defective)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for clean, defective, _ in va_loader:
                clean, defective = clean.to(device), defective.to(device)
                va_losses.append(loss_fn(model(clean), defective).item())

        tr_loss = float(np.mean(tr_losses))
        va_loss = float(np.mean(va_losses))
        history.append({"epoch": epoch, "train_l1": tr_loss, "val_l1": va_loss})

        print(f"Epoch {epoch:03d} | train={tr_loss:.6f} | val={va_loss:.6f}")

        if va_loss < best:
            best, stale = va_loss, 0
            torch.save({"model_state": model.state_dict(), "config": cfg}, checkpoint)
        else:
            stale += 1
            if stale >= int(cfg["training"]["patience"]):
                print("Early stopping.")
                break

    np.save(out / "test_clean.npy", test_pack[0])
    np.save(out / "test_defective.npy", test_pack[1])
    np.save(out / "test_masks.npy", test_pack[2])

    with open(out / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved best checkpoint to {checkpoint}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()
    main(args.config)
