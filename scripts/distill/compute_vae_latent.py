#!/usr/bin/env python3
"""Step 1 of I2V data preparation: encode raw videos into VAE latents.

Port of Self-Forcing-Plus `scripts/compute_vae_latent.py`, adapted to use
MOVA's diffusers-based `AutoencoderKLWan` instead of SFP's `WanVAEWrapper`.

Each video is encoded to a tensor of shape [1, C, T_lat, H_lat, W_lat] and
saved as a .pt dict {prompt_string: latent_tensor}.

Usage:
    torchrun --nproc_per_node=8 scripts/distill/compute_vae_latent.py \
        --ckpt_path /path/to/MOVA-720p \
        --json_path /data/train_data.json \
        --output_latent_folder /data/vae_latents
"""
import sys
import types
import argparse
import glob
import json
import math
import os

import imageio.v3 as iio
import torch
import torch.distributed as dist
import torch.nn.functional as F  # 新增：用于resize
from diffusers.models.autoencoders import AutoencoderKLWan
from tqdm import tqdm

from mova.distill.utils.distributed import launch_distributed_job


@torch.no_grad()
def encode_video(vae, video_tensor, latents_mean, latents_std):
    """Encode [B, C, T, H, W] float bf16 → normalised latent."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = vae.encode(video_tensor).latent_dist.mode()
    # Normalise (diffusers Wan convention)
    mean = torch.tensor(latents_mean, device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
    inv_std = (1.0 / torch.tensor(latents_std, device=z.device, dtype=z.dtype)).view(1, -1, 1, 1, 1)
    return ((z - mean) * inv_std).float().cpu()

# 新增：统一视频尺寸和帧数
def normalize_video(video, target_h=480, target_w=832, target_frames=32):
    """
    标准化视频张量：
    - resize到目标分辨率
    - 截断/补帧到目标帧数
    video: [1, C, T, H, W] (bf16, [-1, 1])
    """
    B, C, T, H, W = video.shape
    
    # 1. Resize (保持比例，pad到目标尺寸，避免拉伸)
    # 计算缩放比例
    scale = min(target_w / W, target_h / H)
    new_h = int(H * scale)
    new_w = int(W * scale)
    # resize
    video_resized = F.interpolate(
        video.view(B*C, T, H, W),  # [B*C, T, H, W]
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    ).view(B, C, T, new_h, new_w)
    # pad到目标尺寸 (上下左右pad)
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    video_padded = F.pad(
        video_resized,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode='constant',
        value=0.0  # pad值为0 (对应归一化后的-1~1中的背景)
    )
    
    # 2. 统一帧数 (截断/补帧)
    if T > target_frames:
        # 截断：取中间帧
        start = (T - target_frames) // 2
        video_framed = video_padded[:, :, start:start+target_frames, :, :]
    else:
        # 补帧：重复最后一帧
        pad_frames = target_frames - T
        pad = video_padded[:, :, -1:, :, :].repeat(1, 1, pad_frames, 1, 1)
        video_framed = torch.cat([video_padded, pad], dim=2)
    
    return video_framed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True, help="MOVA pretrained dir")
    parser.add_argument("--json_path", required=True, help="Path to JSON file containing video_path and caption pairs")
    parser.add_argument("--output_latent_folder", required=True)
    # 新增：可配置目标尺寸/帧数
    parser.add_argument("--target_h", type=int, default=720, help="Target video height (multiple of VAE downsample)")
    parser.add_argument("--target_w", type=int, default=1280, help="Target video width (multiple of VAE downsample)")
    parser.add_argument("--target_frames", type=int, default=121, help="Target video frames")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    launch_distributed_job()
    device = torch.cuda.current_device()
    rank = dist.get_rank()
    world = dist.get_world_size()

    # Load VAE from MOVA checkpoint
    vae = AutoencoderKLWan.from_pretrained(args.ckpt_path, subfolder="video_vae",
                                           torch_dtype=torch.bfloat16).to(device).eval()
    latents_mean = vae.config.latents_mean
    latents_std = vae.config.latents_std

    # Gather (prompt, video_path) pairs from JSON file
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for item in data:
        video_path = item["video_path"]
        caption = item["caption"]
        if os.path.exists(video_path):
            pairs.append((caption, video_path))

    os.makedirs(args.output_latent_folder, exist_ok=True)

    for i in tqdm(range(math.ceil(len(pairs) / world)), disable=rank != 0):
        gi = i * world + rank
        if gi >= len(pairs):
            continue
        prompt, vp = pairs[gi]
        out_path = os.path.join(args.output_latent_folder, f"{gi:08d}.pt")
        if os.path.exists(out_path):
            continue
        try:
            arr = iio.imread(vp)  # [T, H, W, C] uint8
        except Exception as e:
            print(f"[rank {rank}] Failed to read {vp}: {e}")
            continue
        
        # 原始视频张量转换
        video = torch.from_numpy(arr).float().to(device)
        video = video.permute(3, 0, 1, 2).unsqueeze(0) / 255.0  # [1, C, T, H, W]
        video = video * 2 - 1  # 归一化到[-1, 1]
        video = video.to(torch.bfloat16)

        # 新增：标准化视频尺寸和帧数
        video = normalize_video(video, args.target_h, args.target_w, args.target_frames)

        # 编码
        latent = encode_video(vae, video, latents_mean, latents_std)
        # Transpose to SFP convention [B, T_lat, C, H_lat, W_lat]
        latent = latent.permute(0, 2, 1, 3, 4)
        torch.save({prompt: latent}, out_path)

        if gi % 200 == 0 and rank == 0:
            print(f"Processed {gi}/{len(pairs)} | Latent shape: {latent.shape}")

    dist.barrier()
    if rank == 0:
        print(f"Done. Latents written to {args.output_latent_folder}")


if __name__ == "__main__":
    main()