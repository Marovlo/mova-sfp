#!/usr/bin/env bash
# ============================================================
# MOVA I2V DMD distillation — 3-step launch script
# ============================================================
#
# Step 0: Data preprocessing (run once)
#   bash scripts/distill/launch_distill.sh data
#
# Step 1: High-noise distillation
#   bash scripts/distill/launch_distill.sh high
#
# Step 2: Low-noise distillation (needs step 1 checkpoint)
#   bash scripts/distill/launch_distill.sh low
#
# Multi-node: set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT before running.
# ============================================================

set -e

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NPROC=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
RDZV_ID=${RDZV_ID:-mova_distill}

CKPT_PATH=${CKPT_PATH:-/path/to/MOVA-720p}
JSON_PATH=${JSON_PATH:-/data/train_data.json}
LATENT_DIR=${LATENT_DIR:-/data/vae_latents}
LMDB_DIR=${LMDB_DIR:-/data/lmdb_shards}

STEP=${1:?"Usage: $0 {data|high|low}"}

TORCHRUN="torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --nproc_per_node=${NPROC} \
    --rdzv_id=${RDZV_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"

case "${STEP}" in
  data)
    echo "=== Step 0a: Encode videos → VAE latents ==="
    ${TORCHRUN} scripts/distill/compute_vae_latent.py \
        --ckpt_path "${CKPT_PATH}" \
        --json_path "${JSON_PATH}" \
        --output_latent_folder "${LATENT_DIR}"

    echo "=== Step 0b: Build LMDB shards ==="
    python scripts/distill/create_lmdb_shards.py \
        --data_path "${LATENT_DIR}" \
        --json_path "${JSON_PATH}" \
        --lmdb_path "${LMDB_DIR}" \
        --num_shards 16
    echo "=== Data ready at ${LMDB_DIR} ==="
    ;;

  high)
    echo "=== Step 1: High-noise distillation ==="
    ${TORCHRUN} scripts/distill/train_distill.py \
        --config_path configs/distill/mova_distill_i2v_720p_high.yaml \
        --disable-wandb
    ;;

  low)
    echo "=== Step 2: Low-noise distillation ==="
    ${TORCHRUN} scripts/distill/train_distill.py \
        --config_path configs/distill/mova_distill_i2v_720p_low.yaml \
        --disable-wandb
    ;;

  *)
    echo "Unknown step: ${STEP}. Use: data, high, or low."
    exit 1
    ;;
esac
