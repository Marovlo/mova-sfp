#!/usr/bin/env python3
"""Step 2 of I2V data preparation: aggregate per-sample .pt latents + first-frame
images + prompts into multi-shard LMDB databases.

Port of Self-Forcing-Plus `scripts/create_lmdb_14b_shards.py`, adapted to MOVA.

Usage:
    python scripts/distill/create_lmdb_shards.py \
        --data_path /data/vae_latents \
        --prompt_path /data/prompts \
        --video_path /data/videos \
        --lmdb_path /data/lmdb_shards \
        --num_shards 16
"""

import argparse
import glob
import os

import imageio
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

from mova.distill.utils.lmdb_io import process_data_dict, store_arrays_to_lmdb
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Folder with per-sample .pt latents")
    parser.add_argument("--prompt_path", required=True, help="Folder with per-video .txt prompts")
    parser.add_argument("--video_path", required=True, help="Folder with raw video files (for first frame)")
    parser.add_argument("--lmdb_path", required=True, help="Output lmdb directory")
    parser.add_argument("--num_shards", type=int, default=16)
    parser.add_argument("--min_prompt_len", type=int, default=300,
                        help="Skip prompts shorter than this (likely negative prompts)")
    args = parser.parse_args()

    os.makedirs(args.lmdb_path, exist_ok=True)
    map_size = int(1e12)

    envs = []
    for sid in range(args.num_shards):
        path = os.path.join(args.lmdb_path, f"shard_{sid}")
        envs.append(lmdb.open(path, map_size=map_size, subdir=True,
                               readonly=False, metasync=True, sync=True,
                               lock=True, readahead=False, meminit=False))

    # Build prompt → filename map
    prompt_to_fname = {}
    neg_prompts = set()
    for pf in sorted(glob.glob(os.path.join(args.prompt_path, "*.txt"))):
        with open(pf, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if len(txt) < args.min_prompt_len:
            neg_prompts.add(txt)
            continue
        prompt_to_fname[txt] = os.path.basename(pf)

    if neg_prompts:
        print(f"Skipping {len(neg_prompts)} short prompts")

    all_files = sorted(glob.glob(os.path.join(args.data_path, "*.pt")))
    print(f"Found {len(all_files)} .pt files")

    counters = [0] * args.num_shards
    seen = set()
    total = 0
    data_shape = None

    for idx, fpath in tqdm(enumerate(all_files)):
        try:
            raw = torch.load(fpath, map_location="cpu")
            dd = process_data_dict(raw, seen)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            continue

        if dd["latents"].size == 0:
            continue

        # Verify shape consistency
        if data_shape is None:
            data_shape = dd["latents"].shape
        if dd["latents"].shape[1:] != data_shape[1:]:
            continue

        prompt_txt = dd["prompts"][0]
        if len(prompt_txt) < args.min_prompt_len:
            continue

        # Find matching first frame
        if prompt_txt not in prompt_to_fname:
            continue
        video_fname = prompt_to_fname[prompt_txt].replace(".txt", ".mp4")
        video_full = os.path.join(args.video_path, video_fname)
        if not os.path.exists(video_full):
            continue
        try:
            reader = imageio.get_reader(video_full)
            frame = reader.get_data(0)
            reader.close()
            dd["img"] = [Image.fromarray(frame)]
        except Exception as e:
            print(f"Cannot read first frame from {video_full}: {e}")
            continue

        sid = idx % args.num_shards
        store_arrays_to_lmdb(envs[sid], dd, start_index=counters[sid])
        counters[sid] += len(dd["prompts"])
        total += 1

    # Write shape metadata
    for sid, env in enumerate(envs):
        if counters[sid] == 0:
            continue
        with env.begin(write=True) as txn:
            shape_arr = list(data_shape)
            shape_arr[0] = counters[sid]
            txn.put(b"latents_shape", " ".join(map(str, shape_arr)).encode())
            txn.put(b"prompts_shape", f"{counters[sid]}".encode())
            txn.put(b"img_shape", f"{counters[sid]} 480 832 3".encode())

    print(f"Wrote {total} samples into {args.num_shards} shards at {args.lmdb_path}")


if __name__ == "__main__":
    main()
