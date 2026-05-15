# MOVA-SFP: Self-Forcing-Plus DMD Distillation for MOVA

> **将 [Self-Forcing-Plus](https://github.com/GoatWu/Self-Forcing-Plus/tree/wan22) 的 DMD 蒸馏加速训练完整迁移到 [MOVA](https://github.com/OpenMOSS/MOVA) 框架，面向 I2V（图文 → 视频）场景。**

本仓库基于 MOVA 的双塔视频-音频联合扩散架构（`video_dit` + `audio_dit` + `dual_tower_bridge`），将 Wan2.2 MoE 的高/低噪两阶段 DMD 蒸馏无缝接入，实现少步（2-4 step）高质量视频生成。

---

## 核心特性

- **完整 DMD 蒸馏管线**：generator / real_score / fake_score 三角架构，忠实移植 SFP 的 Distribution Matching Distillation (DMD2) 算法
- **双塔感知**：蒸馏 forward 走 MOVA 完整的 `video_dit ↔ audio_dit ↔ dual_tower_bridge` 通路，保证蒸馏分布与 MOVA 推理一致
- **I2V 支持**：接收图片 + 文字作为条件，离线 VAE encode + LMDB 预处理 → 训练时读取首帧 RGB + VAE latent
- **多机 FSDP**：原生 PyTorch FSDP 包裹多模型，支持 8×H100 多节点训练
- **三步清晰流程**：数据预处理 → 高噪蒸馏 → 低噪蒸馏，阶段间可独立调参、验证

---

## 快速开始

### 环境

```bash
pip install -e .[train]
pip install lmdb imageio
```

### 下载预训练模型

```bash
hf download OpenMOSS-Team/MOVA-720p --local-dir ./MOVA-720p
```

### 三步训练

```bash
# 修改路径变量
export CKPT_PATH=./MOVA-720p
export VIDEO_DIR=/data/videos       # 720p mp4 视频
export PROMPT_DIR=/data/prompts     # 同名 .txt prompt
export LATENT_DIR=/data/vae_latents
export LMDB_DIR=/data/lmdb_shards

# Step 0: 离线数据预处理（只做一次）
bash scripts/distill/launch_distill.sh data

# Step 1: 高噪蒸馏
bash scripts/distill/launch_distill.sh high

# Step 2: 低噪蒸馏（依赖 Step 1 的 checkpoint）
bash scripts/distill/launch_distill.sh low
```

### 多机（8 节点 × 8 H100）

```bash
export NNODES=8  NODE_RANK=<0..7>  MASTER_ADDR=<主节点IP>
bash scripts/distill/launch_distill.sh high   # 每个节点分别运行
```

---

## 项目结构

```
mova/distill/                        ← 蒸馏核心代码
  ├── model/
  │   ├── builder.py                 ← 加载 MOVATrain，clone 三份 video DiT
  │   ├── mova_dit_wrapper.py        ← MOVA 双塔 forward wrapper
  │   └── dmd.py                     ← DMD 模型 (generator_loss / critic_loss)
  ├── pipeline/
  │   └── bidirectional_training.py  ← N-step 学生采样器 (backward simulation)
  ├── trainer/
  │   └── distillation_trainer.py    ← 单阶段 trainer
  └── utils/
      ├── distributed.py             ← FSDP wrap + EMA
      ├── scheduler_distill.py       ← FlowMatch add_noise_high/low 增强
      └── lmdb_io.py                 ← LMDB 读写

mova/datasets/
  ├── lmdb_dataset.py                ← I2V: ShardingLMDBDataset
  └── text_prompt_dataset.py         ← T2V: prompt-only 数据集

configs/distill/
  ├── mova_distill_i2v_720p_high.yaml
  └── mova_distill_i2v_720p_low.yaml

scripts/distill/
  ├── compute_vae_latent.py          ← Step 0a: 视频 → VAE latent
  ├── create_lmdb_shards.py          ← Step 0b: latent + 首帧 → LMDB
  ├── train_distill.py               ← 训练入口
  └── launch_distill.sh              ← 一键启动脚本
```

---

## 蒸馏产物

训练完成后得到两组权重：

| 阶段 | 输出路径 | 对应 MOVA 模块 |
|---|---|---|
| Step 1 (高噪) | `logs/.../high/checkpoint_step_XXXXXX/model.pt` | `video_dit` |
| Step 2 (低噪) | `logs/.../low/checkpoint_step_XXXXXX/model.pt` | `video_dit_2` |

加载回 MOVA 推理 pipeline 对应位置即可实现少步采样。

---

## 技术细节

详见 [README_DISTILL.md](./README_DISTILL.md)，包含：

- I2V 数据流完整说明
- 与 SFP 原版的逐文件对应表
- 显存评估与 OOM 逃生方案
- 已知风险与未覆盖项

---

## 致谢

- [Self-Forcing-Plus](https://github.com/GoatWu/Self-Forcing-Plus) — DMD 蒸馏算法与 Wan2.2 MoE 蒸馏方案
- [OpenMOSS/MOVA](https://github.com/OpenMOSS/MOVA) — 双塔视频-音频联合扩散框架
