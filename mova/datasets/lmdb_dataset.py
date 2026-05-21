"""
Sharding LMDB dataset for I2V DMD distillation.

Port of Self-Forcing-Plus `utils/dataset.py:ShardingLMDBDataset`, which stores
pre-encoded VAE latents + prompts + first-frame RGB images in multi-shard lmdb.

Data layout per shard:
  latents_{i}_data  → float16 np array, shape e.g. (1, 21, 16, 60, 104)
  prompts_{i}_data  → utf-8 string
  img_{i}_data      → uint8 np array, shape (H, W, 3)
  latents_shape     → "N 1 21 16 60 104"  (N = #rows in shard)
"""

from __future__ import annotations

import os

import lmdb
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from mova.distill.utils.lmdb_io import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from mova.registry import DATASETS


@DATASETS.register_module()
class ShardingLMDBDataset(Dataset):
    """Read a directory of shard_0/ … shard_N/ LMDB databases."""

    def __init__(self, data_path: str, max_pair: int = int(1e8), transform=None):
        self.envs = []
        self.index = []  # list of (shard_id, local_idx)
        self.img_shapes = []
        self.latents_shape = []
        self.transform = transform

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            if not os.path.isdir(path):
                continue
            env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
            shard_id = len(self.envs)
            self.envs.append(env)

            shape = get_array_shape_from_lmdb(env, "latents")
            self.latents_shape.append(shape)
            img_shape = get_array_shape_from_lmdb(env, "img")
            self.img_shapes.append(img_shape)

            for local_i in range(shape[0]):
                self.index.append((shard_id, local_i))

        self.max_pair = max_pair

    def __len__(self):
        return min(len(self.index), self.max_pair)

    def __getitem__(self, idx):
        shard_id, local_idx = self.index[idx]
        env = self.envs[shard_id]
        shape = self.latents_shape[shard_id]
        img_shape = self.img_shapes[shard_id]
        # print(f"[ShardingLMDBDataset] latent:{shape} img:{img_shape}")

        latents = retrieve_row_from_lmdb(env, "latents", np.float16, local_idx, shape=shape[1:])
        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompt = retrieve_row_from_lmdb(env, "prompts", str, local_idx)

        # First-frame RGB (480×832×3 uint8 → tensor [-1,1])
        try:
            img_np = retrieve_row_from_lmdb(env, "img", np.uint8, local_idx, shape=img_shape[1:])
            img = Image.fromarray(img_np)
            img = TF.to_tensor(img).sub_(0.5).div_(0.5)  # [C, H, W] in [-1,1]
        except KeyError:
            # Fallback: no img stored (pure T2V lmdb). Return zero tensor.
            _, h, w, c = img_shape 
            img = torch.zeros(c, h, w)

        return {
            "prompts": prompt,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
            "img": img,
        }


def lmdb_collate_fn(batch):
    return {
        "prompts": [item["prompts"] for item in batch],
        "ode_latent": torch.stack([item["ode_latent"] for item in batch]),
        "img": torch.stack([item["img"] for item in batch]),
    }
