#!/usr/bin/env python3
"""Step 2 of I2V data preparation: aggregate per-sample .pt latents + first-frame
images + prompts into multi-shard LMDB databases.

Port of Self-Forcing-Plus `scripts/create_lmdb_14b_shards.py`, adapted to MOVA.

Usage:
    python scripts/distill/create_lmdb_shards.py \
        --data_path /data/vae_latents \
        --json_path /data/train_data.json \
        --lmdb_path /data/lmdb_shards \
        --num_shards 16
"""

import argparse
import glob
import json
import os

import imageio
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

from mova.distill.utils.lmdb_io import process_data_dict, store_arrays_to_lmdb
import torch


def resize_first_frame(frame, target_h=480, target_w=832):
    """统一第一帧图片尺寸（与视频标准化尺寸一致）"""
    img = Image.fromarray(frame)
    # 保持比例resize + pad
    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    new_h, new_w = img.size[1], img.size[0]
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    # 创建新画布并粘贴
    new_img = Image.new('RGB', (target_w, target_h), (0, 0, 0))
    new_img.paste(img, (pad_left, pad_top))
    return np.array(new_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Folder with per-sample .pt latents")
    parser.add_argument("--json_path", required=True, help="Path to JSON file containing video_path and caption pairs")
    parser.add_argument("--lmdb_path", required=True, help="Output lmdb directory")
    parser.add_argument("--num_shards", type=int, default=16)
    parser.add_argument("--min_prompt_len", type=int, default=100,
                        help="Skip prompts shorter than this (likely negative prompts)")
    # 新增：与compute_vae_latent.py一致的目标尺寸
    parser.add_argument("--target_h", type=int, default=480, help="Target video height")
    parser.add_argument("--target_w", type=int, default=832, help="Target video width")
    args = parser.parse_args()

    os.makedirs(args.lmdb_path, exist_ok=True)
    map_size = int(1e12)

    envs = []
    for sid in range(args.num_shards):
        path = os.path.join(args.lmdb_path, f"shard_{sid}")
        envs.append(lmdb.open(path, map_size=map_size, subdir=True,
                               readonly=False, metasync=True, sync=True,
                               lock=True, readahead=False, meminit=False))

    # Build prompt → video_path map from JSON file
    prompt_to_fname = {}
    neg_prompts = set()
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        caption = item["caption"]
        video_path = item["video_path"]
        if len(caption) < args.min_prompt_len:
            neg_prompts.add(caption)
            continue
        prompt_to_fname[caption] = video_path

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
            print(f"{fpath} latents size is 0")
            continue

        # 新增：打印形状便于调试
        current_shape = dd["latents"].shape
        if data_shape is None:
            data_shape = current_shape
            print(f"Initial latent shape: {data_shape}")
        else:
            if current_shape[1:] != data_shape[1:]:
                print(f"{fpath} latents shape wrong | Expected: {data_shape[1:]} | Got: {current_shape[1:]}")
                continue

        prompt_txt = dd["prompts"][0]
        print(f"[检查] prompt: {prompt_txt}")
        print(f"是否在字典中: {prompt_txt in prompt_to_fname}")
        if len(prompt_txt) < args.min_prompt_len:
            print(f"{fpath} prompt too short")
            continue

        # Find matching first frame
        if prompt_txt not in prompt_to_fname:
            continue
        video_full = prompt_to_fname[prompt_txt]
        if not os.path.exists(video_full):
            continue
        try:
            reader = imageio.get_reader(video_full)
            frame = reader.get_data(0)
            reader.close()
            # 新增：统一第一帧尺寸
            frame_resized = resize_first_frame(frame, args.target_h, args.target_w)
            dd["img"] = [Image.fromarray(frame_resized)]
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
            # 修改：动态设置img_shape (不再硬编码)
            img_shape = f"{counters[sid]} {args.target_h} {args.target_w} 3"
            txn.put(b"img_shape", img_shape.encode())
            print(f"Shard {sid} | Latents shape: {shape_arr} | Img shape: {img_shape}")

    print(f"Wrote {total} samples into {args.num_shards} shards at {args.lmdb_path}")


if __name__ == "__main__":
    main()