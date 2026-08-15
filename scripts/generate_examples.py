from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import yaml

from src.synthetic import generate_sample

with open("configs/default.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

out = Path("results/examples")
out.mkdir(parents=True, exist_ok=True)

for i in range(6):
    clean, defective, mask, label = generate_sample(
        size=cfg["data"]["volume_size"],
        defect_types=cfg["defects"]["types"],
        min_radius=cfg["defects"]["min_radius"],
        max_radius=cfg["defects"]["max_radius"],
        intensity_min=cfg["defects"]["intensity_min"],
        intensity_max=cfg["defects"]["intensity_max"],
        defect_probability=1.0,
    )

    z = clean.shape[0] // 2

    for name, volume in [("clean", clean), ("defect", defective), ("mask", mask)]:
        plt.figure(figsize=(4, 4))
        plt.imshow(volume[z], cmap="gray")
        plt.title(f"{label} | {name}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out / f"sample_{i}_{name}.png", dpi=160)
        plt.close()

print("Example slices written to results/examples/")
