#!/usr/bin/env python3
"""MOVA DMD distillation entry — one stage per invocation.

Usage:
    # Stage 1: high noise
    torchrun --nproc_per_node=8 scripts/distill/train_distill.py \
        --config_path configs/distill/mova_distill_i2v_720p_high.yaml

    # Stage 2: low noise (after stage 1 checkpoint is ready)
    torchrun --nproc_per_node=8 scripts/distill/train_distill.py \
        --config_path configs/distill/mova_distill_i2v_720p_low.yaml
"""

import argparse
import os

from omegaconf import OmegaConf

# Registry side-effects
import mova.distill.utils.scheduler_distill  # noqa: F401
import mova.datasets.text_prompt_dataset  # noqa: F401
import mova.datasets.lmdb_dataset  # noqa: F401

from mova.distill import DistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--logdir", default="")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--cfg", action="append", default=[],
        help="Override config keys, e.g. --cfg max_steps=10",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    if args.cfg:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.cfg))
    if args.logdir:
        config.logdir = args.logdir
    config.no_save = args.no_save
    if args.disable_wandb:
        config.disable_wandb = True

    os.makedirs(config.logdir, exist_ok=True)

    trainer = DistillationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
