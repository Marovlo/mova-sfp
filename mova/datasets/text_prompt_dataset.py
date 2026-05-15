"""
Text-only prompt datasets for DMD distillation training.

T2V DMD does not require ODE-pair data; it only needs prompt strings to drive
the student/teacher backward simulation (see Self-Forcing-Plus README:
"DMD training for bidirectional models do not need ODE initialization").

Two flavours are provided, mirroring SFP's `utils/dataset.py`:
  * TextFolderDataset : a directory where each `*.txt` file contains one prompt.
  * TextFileDataset   : a single text file with one prompt per line.
"""

from __future__ import annotations

import os
from typing import List

from torch.utils.data import Dataset

from mova.registry import DATASETS


@DATASETS.register_module()
class TextFolderDataset(Dataset):
    def __init__(self, data_path: str, max_count: int = 200000, transform=None):
        self.data_path = data_path
        self.transform = transform  # accepted but unused (kept for builder API parity)
        self.texts: List[str] = []
        for fname in sorted(os.listdir(data_path)):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(data_path, fname), "r", encoding="utf-8") as f:
                self.texts.append(f.read().strip())
            if len(self.texts) >= max_count:
                break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"prompt": self.texts[idx], "idx": idx}


@DATASETS.register_module()
class TextFileDataset(Dataset):
    def __init__(self, prompt_path: str, transform=None):
        self.transform = transform
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.texts = [line.rstrip() for line in f if line.strip()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"prompt": self.texts[idx], "idx": idx}


def text_collate_fn(batch):
    return {
        "prompts": [item["prompt"] for item in batch],
        "idx": [item["idx"] for item in batch],
    }
