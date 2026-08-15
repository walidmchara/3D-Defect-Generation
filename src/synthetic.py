from __future__ import annotations

import math
import random
import numpy as np


def make_base_volume(size: int) -> np.ndarray:
    """Create a smooth CT-like synthetic material block."""
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    c = (size - 1) / 2.0
    r = size * 0.42
    body = ((x-c)**2 + (y-c)**2 + (z-c)**2) <= r**2

    volume = np.zeros((size, size, size), dtype=np.float32)
    volume[body] = 0.65

    # mild spatial heterogeneity
    gradient = 0.08 * (z / max(size - 1, 1))
    volume[body] += gradient[body].astype(np.float32)

    noise = np.random.normal(0.0, 0.02, size=volume.shape).astype(np.float32)
    volume += noise
    return np.clip(volume, 0.0, 1.0)


def _random_center(size, margin=6):
    lo = min(margin, size // 4)
    hi = max(lo + 1, size - lo)
    return [random.randint(lo, hi - 1) for _ in range(3)]


def add_sphere(volume, mask, radius, intensity):
    size = volume.shape[0]
    cz, cy, cx = _random_center(size)
    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    defect = ((x-cx)**2 + (y-cy)**2 + (z-cz)**2) <= radius**2
    volume[defect] = np.clip(volume[defect] + intensity, 0.0, 1.0)
    mask[defect] = 1.0


def add_ellipsoid(volume, mask, radius, intensity):
    size = volume.shape[0]
    cz, cy, cx = _random_center(size)
    rz = max(1, int(radius * random.uniform(0.6, 1.5)))
    ry = max(1, int(radius * random.uniform(0.6, 1.5)))
    rx = max(1, int(radius * random.uniform(0.6, 1.5)))

    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    defect = (
        ((x-cx)/rx)**2
        + ((y-cy)/ry)**2
        + ((z-cz)/rz)**2
    ) <= 1.0

    volume[defect] = np.clip(volume[defect] + intensity, 0.0, 1.0)
    mask[defect] = 1.0


def add_crack(volume, mask, radius, intensity):
    size = volume.shape[0]
    cz, cy, cx = _random_center(size)

    normal = np.random.normal(size=3)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    z, y, x = np.mgrid[0:size, 0:size, 0:size]
    dz, dy, dx = z-cz, y-cy, x-cx

    plane_distance = np.abs(
        normal[0]*dz + normal[1]*dy + normal[2]*dx
    )

    radial = np.sqrt(dx**2 + dy**2 + dz**2)
    defect = (plane_distance <= 0.8) & (radial <= radius * 1.8)

    volume[defect] = np.clip(volume[defect] + intensity, 0.0, 1.0)
    mask[defect] = 1.0


def generate_sample(size, defect_types, min_radius, max_radius,
                    intensity_min, intensity_max, defect_probability):
    clean = make_base_volume(size)
    defective = clean.copy()
    mask = np.zeros_like(clean, dtype=np.float32)

    defect_name = "none"
    if random.random() < defect_probability:
        defect_name = random.choice(defect_types)
        radius = random.randint(min_radius, max_radius)
        intensity = random.uniform(intensity_min, intensity_max)

        if defect_name == "sphere":
            add_sphere(defective, mask, radius, intensity)
        elif defect_name == "ellipsoid":
            add_ellipsoid(defective, mask, radius, intensity)
        elif defect_name == "crack":
            add_crack(defective, mask, radius, intensity)
        else:
            raise ValueError(f"Unknown defect type: {defect_name}")

    return clean.astype(np.float32), defective.astype(np.float32), mask, defect_name
