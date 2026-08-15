# 3D Defect Generation

Generative AI framework for realistic **3D volumetric defect synthesis** and industrial inspection research.

This repository provides a research pipeline for creating synthetic volumetric defect data, training a 3D representation model, and evaluating generated or reconstructed defect volumes.

## Project Goals

- Generate synthetic 3D volumetric defects for industrial inspection.
- Build reusable data-generation and training pipelines.
- Learn 3D representations with a 3D autoencoder.
- Evaluate generated and reconstructed volumes.
- Provide a foundation for future diffusion, VAE, GAN, Transformer, and physics-informed models.

## Repository Structure

```text
3D-Defect-Generation/
├── configs/
├── data/
├── results/
├── scripts/
├── src/
│   ├── dataset.py
│   ├── synthetic.py
│   ├── train.py
│   ├── evaluate.py
│   └── models/
│       └── autoencoder3d.py
├── README_V1_RUN.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Current Model

The repository contains a **3D Autoencoder** implementation for volumetric representation learning. It provides a baseline for reconstruction-based anomaly analysis and a building block for more advanced generative architectures.

## Synthetic Defect Pipeline

```text
3D Volume
    |
    v
Synthetic Defect Generation
    |
    v
Dataset Preparation
    |
    v
3D Autoencoder
    |
    v
Reconstruction / Latent Representation
    |
    v
Evaluation and Visualization
```

The `src/synthetic.py` module contains the synthetic-volume generation logic, while `src/dataset.py`, `src/train.py`, and `src/evaluate.py` support the machine-learning workflow.

## Installation

```bash
git clone https://github.com/walidmchara/3D-Defect-Generation.git
cd 3D-Defect-Generation
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\\Scripts\\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

See `README_V1_RUN.md` for the current execution workflow and experiment notes.

Typical workflow:

1. Prepare or generate volumetric data.
2. Configure the experiment.
3. Train the 3D model.
4. Generate or reconstruct defect volumes.
5. Evaluate and inspect the results.

## Research Directions

- 3D Variational Autoencoders
- 3D GAN-based defect synthesis
- Latent diffusion models for volumetric CT data
- Conditional generation by defect type, size, severity, and location
- Physics-informed defect constraints
- Multi-scale defect synthesis
- Synthetic-to-real domain adaptation
- Scanner, noise, and reconstruction artifact simulation
- Few-shot anomaly detection using synthetic defects
- Segmentation-aware defect generation

## Industrial Applications

- Battery CT inspection
- Electronics and semiconductor inspection
- Additive-manufacturing defect analysis
- Structural component inspection
- Porosity and crack synthesis
- Non-destructive testing
- Synthetic training-data generation for rare defects

## Why Synthetic 3D Defects?

Real industrial defects are often rare, expensive to acquire, difficult to reproduce, and dependent on material, scanner, and reconstruction conditions. Controlled synthetic generation can help create larger and more diverse training datasets while allowing defect geometry, location, scale, and severity to be varied systematically.

## Technology Stack

The project is organized as a Python research repository with modular code for dataset handling, synthetic generation, model training, and evaluation. See `requirements.txt` for the exact dependencies used by the current implementation.

## Citation

If you use or extend this repository in academic work, please cite the repository and associated publication when available.

## Author

**Walid Mchara**

Research interests: Artificial Intelligence, Deep Learning, Generative AI, 3D volumetric modeling, predictive maintenance, battery health, and intelligent energy systems.

## License

This project is distributed under the license included in the `LICENSE` file.

---

### Keywords

`3D defect generation` · `generative AI` · `volumetric deep learning` · `3D autoencoder` · `industrial inspection` · `computed tomography` · `CT` · `X-ray` · `anomaly detection` · `synthetic data` · `non-destructive testing`
