"""LMDB read/write helpers — direct port from Self-Forcing-Plus `utils/lmdb.py`."""

from __future__ import annotations

import numpy as np


def get_array_shape_from_lmdb(env, array_name: str):
    with env.begin() as txn:
        raw = txn.get(f"{array_name}_shape".encode())
        if raw is None:
            raise KeyError(f"Shape key '{array_name}_shape' not found in lmdb")
        return tuple(map(int, raw.decode().split()))


def store_arrays_to_lmdb(env, arrays_dict: dict, start_index: int = 0):
    with env.begin(write=True) as txn:
        for array_name, array in arrays_dict.items():
            for i, row in enumerate(array):
                if isinstance(row, str):
                    row_bytes = row.encode()
                else:
                    row_bytes = row.tobytes()
                txn.put(f"{array_name}_{start_index + i}_data".encode(), row_bytes)


def retrieve_row_from_lmdb(lmdb_env, array_name: str, dtype, row_index: int, shape=None):
    data_key = f"{array_name}_{row_index}_data".encode()
    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)
    if row_bytes is None:
        raise KeyError(f"Key {data_key} not found in lmdb")
    if dtype == str:
        return row_bytes.decode()
    array = np.frombuffer(row_bytes, dtype=dtype)
    if shape is not None and len(shape) > 0:
        array = array.reshape(shape)
    return array


def process_data_dict(data_dict: dict, seen_prompts: set):
    all_videos, all_prompts = [], []
    for prompt, video in data_dict.items():
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        all_videos.append(video.half().numpy())
        all_prompts.append(prompt)
    if not all_videos:
        return {"latents": np.array([]), "prompts": np.array([])}
    return {
        "latents": np.concatenate(all_videos, axis=0),
        "prompts": np.array(all_prompts),
    }
