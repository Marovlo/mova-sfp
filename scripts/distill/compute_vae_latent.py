#!/usr/bin/env python3
"""Step 1 of I2V data preparation: encode raw videos into VAE latents.

Port of Self-Forcing-Plus `scripts/compute_vae_latent.py`, adapted to use
MOVA's diffusers-based `AutoencoderKLWan` instead of SFP's `WanVAEWrapper`.

Each video is encoded to a tensor of shape [1, C, T_lat, H_lat, W_lat] and
saved as a .pt dict {prompt_string: latent_tensor}.

Usage:
    torchrun --nproc_per_node=8 scripts/distill/compute_vae_latent.py \
        --ckpt_path /path/to/MOVA-720p \
        --input_video_folder /data/videos \
        --prompt_folder /data/prompts \
        --output_latent_folder /data/vae_latents
"""

import argparse
import glob
import math
import os

import imageio.v3 as iio
import torch
import torch.distributed as dist
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True, help="MOVA pretrained dir")
    parser.add_argument("--input_video_folder", required=True)
    parser.add_argument("--prompt_folder", required=True)
    parser.add_argument("--output_latent_folder", required=True)
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

    # Gather (prompt, video_path) pairs
    exts = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
    video_files = []
    for ext in exts:
        video_files.extend(glob.glob(os.path.join(args.input_video_folder, ext)))
    video_files.sort()

    pairs = []
    for vf in video_files:
        stem = os.path.splitext(os.path.basename(vf))[0]
        pf = os.path.join(args.prompt_folder, stem + ".txt")
        if os.path.exists(pf):
            with open(pf, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            pairs.append((prompt, vf))

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
            arr = iio.imread(vp, plugin="pyav")  # [T, H, W, C] uint8
        except Exception as e:
            print(f"[rank {rank}] Failed to read {vp}: {e}")
            continue
        video = torch.from_numpy(arr).float().to(device)
        video = video.permute(3, 0, 1, 2).unsqueeze(0) / 255.0  # [1, C, T, H, W]
        video = video * 2 - 1
        video = video.to(torch.bfloat16)

        latent = encode_video(vae, video, latents_mean, latents_std)
        # Transpose to SFP convention [B, T_lat, C, H_lat, W_lat]
        latent = latent.permute(0, 2, 1, 3, 4)
        torch.save({prompt: latent}, out_path)

        if gi % 200 == 0 and rank == 0:
            print(f"Processed {gi}/{len(pairs)}")

    dist.barrier()
    if rank == 0:
        print(f"Done. Latents written to {args.output_latent_folder}")


if __name__ == "__main__":
    main()
