# MOVA × Self-Forcing-Plus DMD 蒸馏（I2V，迁移说明）

将 [Self-Forcing-Plus (`wan22` 分支)](https://github.com/GoatWu/Self-Forcing-Plus/tree/wan22)
的 DMD 蒸馏训练完整迁移到 [OpenMOSS/MOVA](https://github.com/OpenMOSS/MOVA)，
面向 **I2V（图文→视频）** 场景，保留 SFP 原版的三步流程结构：

```
Step 0: 离线数据预处理（视频 → VAE latent → LMDB）
Step 1: 高噪蒸馏（video_dit）
Step 2: 低噪蒸馏（video_dit_2，依赖 Step 1 产物）
```

---

## 1. 三步流程设计

### 为什么保留离线数据预处理？

I2V 蒸馏训练时每个 batch 需要 `(prompt, vae_latent, first_frame_rgb)`。VAE encode
720p×81f 视频是确定性、无梯度、计算密集的操作。把它放到训练 loop 里会：

- 消耗 video_vae 显存（与 DMD 的 5 模型 FSDP 竞争）
- 每 step 增加数秒 VAE forward，训练吞吐下降一个数量级
- 视频 IO jitter 导致 FSDP allreduce 尾延迟

这是 video diffusion 领域的标准最佳实践（Stable Video、Wan 都离线预 encode）。

### 为什么拆成两条训练命令（不合并 high→low）？

- 阶段间可以手动检查 checkpoint 质量
- 可以独立调参（两阶段的 lr、CFG、max_steps 可能不同）
- 避免单进程内两次 FSDP wrap/teardown 的 CUDA 碎片问题
- 多机场景 stage1→stage2 权重交接用落盘 `.pt` 而非内存直传，鲁棒

---

## 2. 文件清单

```
mova/distill/                               ← 蒸馏核心
  ├── model/
  │   ├── builder.py                        ← 加载 MOVATrain，clone 三份 video DiT
  │   ├── mova_dit_wrapper.py               ← MOVA 双塔 forward wrapper（替代 SFP WanDiffusionWrapper）
  │   └── dmd.py                            ← DMD 模型（generator_loss / critic_loss）
  ├── pipeline/
  │   └── bidirectional_training.py         ← N-step 学生采样器（backward simulation）
  ├── trainer/
  │   └── distillation_trainer.py           ← 单阶段 trainer（--stage 由配置文件决定）
  └── utils/
      ├── distributed.py                    ← FSDP wrap + EMA_FSDP
      ├── scheduler_distill.py              ← FlowMatchScheduler 增强（add_noise_high/low）
      └── lmdb_io.py                        ← LMDB 读写工具

mova/datasets/
  ├── text_prompt_dataset.py                ← T2V 用 prompt-only 数据集
  └── lmdb_dataset.py                       ← I2V 用 ShardingLMDBDataset

configs/distill/
  ├── mova_distill_i2v_720p_high.yaml       ← 高噪配置
  └── mova_distill_i2v_720p_low.yaml        ← 低噪配置

scripts/distill/
  ├── compute_vae_latent.py                 ← Step 0a: 视频 → VAE latent（分布式）
  ├── create_lmdb_shards.py                 ← Step 0b: latent+首帧 → LMDB
  ├── train_distill.py                      ← Step 1/2: 训练入口
  └── launch_distill.sh                     ← 一键启动脚本（data / high / low）
```

---

## 3. 运行方式

### 3.1 环境

```bash
cd MOVA
pip install -e .[train]
pip install lmdb imageio
```

### 3.2 下载预训练模型

```bash
hf download OpenMOSS-Team/MOVA-720p --local-dir ./MOVA-720p
```

### 3.3 准备原始数据

```
/data/videos/      ← 720p mp4 视频文件
/data/prompts/     ← 与视频同名的 .txt prompt 文件
```

### 3.4 执行三步

```bash
# 修改 launch_distill.sh 头部的路径变量，或 export:
export CKPT_PATH=./MOVA-720p
export VIDEO_DIR=/data/videos
export PROMPT_DIR=/data/prompts
export LATENT_DIR=/data/vae_latents
export LMDB_DIR=/data/lmdb_shards

# Step 0: 数据预处理（离线，只做一次）
bash scripts/distill/launch_distill.sh data

# Step 1: 高噪蒸馏
bash scripts/distill/launch_distill.sh high

# Step 2: 低噪蒸馏（确认 high 阶段 checkpoint 路径在 low yaml 里配对）
bash scripts/distill/launch_distill.sh low
```

### 3.5 多机（8 节点 × 8 H100）

```bash
# 每个节点：
export NNODES=8
export NODE_RANK=<0..7>
export MASTER_ADDR=<主节点IP>
export MASTER_PORT=29500
bash scripts/distill/launch_distill.sh high  # 或 low
```

### 3.6 产物

- `logs/mova_distill_i2v_720p_high/checkpoint_step_XXXXXX/model.pt` → 蒸馏后的 `video_dit`
- `logs/mova_distill_i2v_720p_low/checkpoint_step_XXXXXX/model.pt`  → 蒸馏后的 `video_dit_2`

加载回 MOVA 推理 pipeline 的对应位置即可验证少步采样质量。

---

## 4. 与 SFP 原版对应

| SFP 原文件 | MOVA 迁移版本 |
|---|---|
| `scripts/compute_vae_latent.py` | `scripts/distill/compute_vae_latent.py`（适配 diffusers VAE） |
| `scripts/create_lmdb_14b_shards.py` | `scripts/distill/create_lmdb_shards.py` |
| `utils/dataset.py:ShardingLMDBDataset` | `mova/datasets/lmdb_dataset.py` |
| `utils/lmdb.py` | `mova/distill/utils/lmdb_io.py` |
| `train.py` | `scripts/distill/train_distill.py` |
| `trainer/distillation.py` | `mova/distill/trainer/distillation_trainer.py` |
| `model/dmd.py` + `model/base.py` | `mova/distill/model/dmd.py` |
| `pipeline/bidirectional_training.py` | `mova/distill/pipeline/bidirectional_training.py` |
| `utils/wan_wrapper.py` | `mova/distill/model/mova_dit_wrapper.py` |
| `utils/scheduler.py` | `mova/distill/utils/scheduler_distill.py` |
| `utils/distributed.py` | `mova/distill/utils/distributed.py` |
| `configs/wan22_high_i2v.yaml` | `configs/distill/mova_distill_i2v_720p_high.yaml` |
| `configs/wan22_low_i2v.yaml` | `configs/distill/mova_distill_i2v_720p_low.yaml` |

---

## 5. 关键技术说明

### 5.1 I2V 数据流

```
[离线]
  视频(720p mp4) → compute_vae_latent.py → per-sample .pt {prompt: latent[1,21,16,90,160]}
  .pt + 首帧RGB → create_lmdb_shards.py → lmdb shards (latents + prompts + img)

[训练时]
  ShardingLMDBDataset → batch:
    prompt             : str
    ode_latent         : [B, 1, 21, 16, H_lat, W_lat]
    img                : [B, C, H, W] (首帧 RGB, [-1,1])

  Trainer:
    initial_latent = ode_latent[:, -1][:, 0:1]  # 首帧 clean VAE latent
    y = video_vae.encode(first_frame) + mask     # MOVA 20-ch condition
    → 传给 DMD generator_loss / critic_loss
```

### 5.2 为什么不蒸馏 audio_dit

SFP 原版只蒸馏 video DiT。audio_dit 和 dual_tower_bridge 在训练时冻结参与
forward，保证 cross-attention 分布与 MOVA 推理一致，但不接收梯度。

### 5.3 显存估算（8×H100 80GB / 机）

DMD 同驻：
- generator `video_dit` + `video_dit_2`（trainable, FSDP）
- fake_score `video_dit` + `video_dit_2`（trainable, FSDP）
- real_score `video_dit` + `video_dit_2`（frozen, FSDP）
- high_noise_teacher `video_dit`（仅 low 阶段, frozen, FSDP）
- `audio_dit`（frozen, GPU, ~1.3B）
- `dual_tower_bridge`（frozen, GPU）
- `text_encoder`（CPU offload）
- `video_vae`（I2V 时 GPU，仅首帧 encode 时用）

如果 8 卡 OOM，依次尝试：
1. `gradient_checkpointing_offload: true`
2. 增加节点数（16 卡 → per-rank shard 减半）
3. 缩减 `num_training_frames` 到 13 或 9 做 dry-run

### 5.4 boundary_step 差异

- SFP T2V `wan22_high.yaml`: boundary_step = 500
- SFP I2V `wan22_high_i2v.yaml`: boundary_step = 500
- MOVA 原始训练: boundary_ratio = 0.9 → boundary_step ≈ 900

**当前配置使用 SFP I2V 的 boundary_step=500**，与蒸馏算法对齐。如果你的 MOVA
720p checkpoint 是按 boundary=900 训的，蒸馏用 500 不会冲突——boundary 只影响
student 去噪范围的切分，teacher 权重不变。

---

## 6. 已知风险

1. **LMDB 首帧分辨率硬编码**：`lmdb_dataset.py` 默认 480×832（SFP I2V 原值）。
   如果你的 MOVA-720p 首帧是 720×1280，需要改 `create_lmdb_shards.py` 里的 shape
   以及 `lmdb_dataset.py:retrieve_row_from_lmdb(..., shape=(720,1280,3))`。
2. **`paired_timesteps` 未用于蒸馏**：MOVA 推理时 visual/audio 可能有错位 timestep
   pair；DMD 训练时 visual/audio 用同一个 timestep id。
3. **未测试端到端运行**：Mac 无 GPU / 分布式，只做了语法检查 + 静态引用扫描。
   建议先以极小配置（`image_or_video_shape: [1,5,16,16,16]`, `max_steps: 1`）
   在 1 卡上 dry-run。
