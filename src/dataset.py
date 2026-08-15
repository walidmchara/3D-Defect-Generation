from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class VolumePairDataset(Dataset):
    def __init__(self, clean, defective, masks):
        self.clean = torch.as_tensor(clean[:, None], dtype=torch.float32)
        self.defective = torch.as_tensor(defective[:, None], dtype=torch.float32)
        self.masks = torch.as_tensor(masks[:, None], dtype=torch.float32)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        return self.clean[index], self.defective[index], self.masks[index]
