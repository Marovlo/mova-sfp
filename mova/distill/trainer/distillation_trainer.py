"""
MOVA Distillation Trainer — I2V-aware, single-stage-per-launch.

Each invocation trains ONE stage (high_noise or low_noise) via --stage.
The low_noise stage requires a pre-distilled high-noise checkpoint
(`high_noise_distill_ckpt` in config, the .pt produced by the high stage).

I2V data flow:
  * Offline: videos → compute_vae_latent.py → .pt → create_lmdb_shards.py → lmdb
  * Training: lmdb → (prompt, vae_latent, first_frame_rgb) per batch
    - first_frame_rgb → video_vae encode → y (20-ch condition)
    - vae_latent[:, -1][:, 0:1] → initial_latent (first-frame clean latent for I2V noise)
"""

from __future__ import annotations

import gc
import os
import time
import logging
from tqdm import tqdm
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, distributed as torch_dist_data

from mova.distill.model.dmd import MOVADMD
from mova.distill.model.builder import build_distill_modules, encode_text_prompts
from mova.distill.model.mova_dit_wrapper import MOVAVideoDiTWrapper
from mova.diffusion.models.wan_video_dit import DiTBlock as WanDiTBlock
from mova.distill.utils.distributed import (
    EMA, EMA_FSDP, fsdp_state_dict, fsdp_wrap, launch_distributed_job,
    build_device_mesh,
)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


class DistillationTrainer:
    """Single-stage (high or low) distillation driver.

    Usage:
        torchrun ... train_distill.py --config ... --stage high
        torchrun ... train_distill.py --config ... --stage low
    """

    @staticmethod
    def _move_module_to_device(module, device):
        module.to(device)
        with torch.no_grad():
            for p in module.parameters():
                p.data = p.data.to(device=device)
            for b in module.buffers():
                b.data = b.data.to(device=device)

    def __init__(self, config):
        self.config = config
        self.step = 0

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        launch_distributed_job()
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main = self.global_rank == 0
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32

        if config.seed == 0:
            seed = torch.randint(0, 10_000_000, (1,), device=self.device)
            dist.broadcast(seed, src=0)
            config.seed = int(seed.item())
        torch.manual_seed(config.seed + self.global_rank)

        self.i2v = getattr(config, "i2v", False)
        self.disable_wandb = getattr(config, "disable_wandb", True)

        dp_size = getattr(config, "dp_size", 1)
        cp_size = getattr(config, "cp_size", 1)
        if self.world_size > 1 and (dp_size > 1 or cp_size > 1):
            self.mesh, self._dp_size, self._cp_size, self._fsdp_size = \
                build_device_mesh(dp_size=dp_size, cp_size=cp_size)
            self.cp_mesh = self.mesh["cp"] if cp_size > 1 else None
        else:
            self.mesh = None
            self._dp_size = 1
            self._cp_size = 1
            self._fsdp_size = self.world_size
            self.cp_mesh = None

        if self.cp_mesh is not None:
            from yunchang import set_seq_parallel_pg
            cp_sz = self._cp_size
            MAX_ULYSSES = 4
            sp_ulysses = min(MAX_ULYSSES, cp_sz)
            sp_ring = cp_sz // sp_ulysses
            assert sp_ring * sp_ulysses == cp_sz
            set_seq_parallel_pg(
                sp_ulysses, sp_ring,
                dist.get_rank(), dist.get_world_size(),
                use_ulysses_low=True,
            )
            if self.is_main:
                print(f"[CP] ulysses={sp_ulysses}, ring={sp_ring}")

        if self.is_main:
            print(f"[Distill] stage={config.training_target}, i2v={self.i2v}")
            print(f"[Distill] Building modules from {config.pretrained_path}")

        self._modules = build_distill_modules(
            pretrained_path=config.pretrained_path,
            training_target=config.training_target,
            use_gradient_checkpointing=config.gradient_checkpointing,
            use_gradient_checkpointing_offload=getattr(config, "gradient_checkpointing_offload", False),
            torch_dtype=self.dtype,
        )

        self._setup()

    def _setup(self):
        cfg = self.config
        target = cfg.training_target
        m = self._modules

        high_noise_teacher = None
        if target == "low_noise":
            import copy
            distill_path = cfg.high_noise_distill_ckpt
            if self.is_main:
                print(f"[Distill] Loading high-noise distilled ckpt: {distill_path}")
            teacher_dit_high = copy.deepcopy(m.real_dit)
            state = torch.load(distill_path, map_location="cpu")
            if "generator" in state:
                state = state["generator"]
            teacher_dit_high.load_state_dict(state, strict=True)
            teacher_dit_high.requires_grad_(False).eval()

            high_noise_teacher = MOVAVideoDiTWrapper(
                video_dit=teacher_dit_high,
                target="high_noise",
                boundary_step=cfg.boundary_step,
                timestep_shift=cfg.timestep_shift,
                audio_dit=m.audio_dit,
                dual_tower_bridge=m.dual_tower_bridge,
                scheduler=m.scheduler,
                use_gradient_checkpointing=m.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=m.use_gradient_checkpointing_offload,
            )

        def _mk(dit, tgt):
            return MOVAVideoDiTWrapper(
                video_dit=dit,
                target=tgt,
                boundary_step=cfg.boundary_step,
                timestep_shift=cfg.timestep_shift,
                audio_dit=m.audio_dit,
                dual_tower_bridge=m.dual_tower_bridge,
                scheduler=m.scheduler,
                use_gradient_checkpointing=m.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=m.use_gradient_checkpointing_offload,
            )

        generator = _mk(m.generator_dit, target)
        real_score = _mk(m.real_dit, target)
        fake_score = _mk(m.fake_dit, target)
        for p in real_score.parameters():
            p.requires_grad_(False)

        wrap_kw = dict(
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            wrap_strategy=cfg.fsdp_wrap_strategy,
            transformer_module={WanDiTBlock} if cfg.fsdp_wrap_strategy == "transformer" else None,
            device_mesh=self.mesh,
        )
        
        def _count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        def _param_memory_gb(module):
            total_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
            return total_bytes / 1e9
        
        gen_params_full = _count_params(generator.video_dit)
        fake_params_full = _count_params(fake_score.video_dit)
        real_params_full = _count_params(real_score.video_dit)
        total_params_full = gen_params_full + fake_params_full + real_params_full
        
        if self.is_main:
            print(f"\n{'='*70}")
            print(f"[FSDP VERIFICATION] Configuration:")
            print(f"  world_size: {self.world_size}")
            print(f"  dp_size: {self._dp_size}, cp_size: {self._cp_size}, fsdp_size: {self._fsdp_size}")
            print(f"  mesh: {self.mesh}")
            print(f"  cp_mesh: {self.cp_mesh}")
            print(f"  sharding_strategy: {cfg.sharding_strategy}")
            print(f"  mixed_precision: {cfg.mixed_precision}")
            print(f"  wrap_strategy: {cfg.fsdp_wrap_strategy}")
            
            if self._dp_size > 1:
                print(f"\n[DP VERIFICATION] Data Parallel Configuration:")
                print(f"  dp_size: {self._dp_size}")
                print(f"  Effective batch_size per DP group: {getattr(cfg, 'batch_size', 1) * self._dp_size}")
                if self.mesh is not None:
                    try:
                        dp_mesh = self.mesh["dp"]
                        print(f"  dp_mesh: {dp_mesh}")
                        print(f"  DP ranks: {dp_mesh.mesh.tolist() if hasattr(dp_mesh.mesh, 'tolist') else dp_mesh.mesh}")
                    except KeyError:
                        print(f"  dp_mesh: Not found in device mesh")
                print(f"  ✓ DP enabled: Each rank processes different data, gradients synchronized across DP ranks")
            else:
                print(f"\n[DP VERIFICATION] DP disabled (dp_size=1)")
            
            print(f"\n[FSDP VERIFICATION] FULL Model Size (Before FSDP):")
            print(f"  generator.video_dit: {gen_params_full/1e6:.2f}M params ({_param_memory_gb(generator.video_dit):.2f} GB)")
            print(f"  fake_score.video_dit: {fake_params_full/1e6:.2f}M params ({_param_memory_gb(fake_score.video_dit):.2f} GB)")
            print(f"  real_score.video_dit: {real_params_full/1e6:.2f}M params ({_param_memory_gb(real_score.video_dit):.2f} GB)")
            print(f"  TOTAL: {total_params_full/1e6:.2f}M params ({_param_memory_gb(generator.video_dit)+_param_memory_gb(fake_score.video_dit)+_param_memory_gb(real_score.video_dit):.2f} GB)")
            print(f"{'='*70}\n")
        
        generator.video_dit = fsdp_wrap(generator.video_dit, cpu_offload=True, **wrap_kw)
        generator.model = generator.video_dit

        fake_score.video_dit = fsdp_wrap(fake_score.video_dit, cpu_offload=True, **wrap_kw)
        fake_score.model = fake_score.video_dit

        real_score.video_dit = fsdp_wrap(real_score.video_dit, cpu_offload=True, **wrap_kw)
        real_score.model = real_score.video_dit

        if high_noise_teacher is not None:
            high_noise_teacher.video_dit = fsdp_wrap(
                high_noise_teacher.video_dit, cpu_offload=False, **wrap_kw,
            )
        
        gen_params_sharded = _count_params(generator.video_dit)
        fake_params_sharded = _count_params(fake_score.video_dit)
        real_params_sharded = _count_params(real_score.video_dit)
        total_sharded = gen_params_sharded + fake_params_sharded + real_params_sharded
        
        if self.is_main:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            print(f"\n{'='*70}")
            print(f"[FSDP VERIFICATION] SHARDED Model Size (After FSDP, rank {self.global_rank}):")
            print(f"  generator.video_dit: {gen_params_sharded/1e6:.2f}M params")
            print(f"  fake_score.video_dit: {fake_params_sharded/1e6:.2f}M params")
            print(f"  real_score.video_dit: {real_params_sharded/1e6:.2f}M params")
            print(f"  TOTAL on this rank: {total_sharded/1e6:.2f}M params")
            print(f"\n[FSDP VERIFICATION] FSDP Wrapping Status:")
            print(f"  generator.video_dit is FSDP: {isinstance(generator.video_dit, FSDP)}")
            print(f"  fake_score.video_dit is FSDP: {isinstance(fake_score.video_dit, FSDP)}")
            print(f"  real_score.video_dit is FSDP: {isinstance(real_score.video_dit, FSDP)}")
            print(f"\n[FSDP VERIFICATION] Memory Reduction:")
            if total_params_full > 0:
                reduction_ratio = total_params_full / total_sharded if total_sharded > 0 else 0
                print(f"  Expected reduction factor: ~{self._fsdp_size}x (fsdp_size)")
                print(f"  Actual reduction factor: {reduction_ratio:.2f}x")
                print(f"  Sharding efficiency: {reduction_ratio/self._fsdp_size*100:.1f}%")
            print(f"{'='*70}\n")

        text_offload = getattr(cfg, "text_encoder_cpu_offload", True)
        self._move_module_to_device(
            m.text_encoder, "cpu" if text_offload else self.device
        )
        m.text_encoder.requires_grad_(False)
        if self.i2v:
            self._move_module_to_device(m.video_vae, self.device)
        else:
            self._move_module_to_device(m.video_vae, "cpu")
        m.video_vae.requires_grad_(False)
        self._move_module_to_device(m.audio_vae, "cpu")
        m.audio_vae.requires_grad_(False)

        audio_dit_offload = getattr(cfg, "audio_dit_cpu_offload", True)
        bridge_offload = getattr(cfg, "dual_tower_bridge_cpu_offload", True)
        self._audio_dit_offload = audio_dit_offload
        self._bridge_offload = bridge_offload
        self._move_module_to_device(
            m.audio_dit, "cpu" if audio_dit_offload else self.device
        )
        self._move_module_to_device(
            m.dual_tower_bridge, "cpu" if bridge_offload else self.device
        )
        m.audio_dit.requires_grad_(False)
        m.dual_tower_bridge.requires_grad_(False)

        self.model = MOVADMD(
            config=cfg,
            device=self.device,
            generator=generator,
            real_score=real_score,
            fake_score=fake_score,
            scheduler=m.scheduler,
            shared_audio_dit=m.audio_dit,
            shared_dual_tower_bridge=m.dual_tower_bridge,
            high_noise_teacher=high_noise_teacher,
        )

        gen_params = [p for p in self.model.generator.parameters() if p.requires_grad]
        crit_params = [p for p in self.model.fake_score.parameters() if p.requires_grad]
        self.generator_optimizer = torch.optim.AdamW(
            gen_params, lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=getattr(cfg, "weight_decay", 0.01),
        )
        self.critic_optimizer = torch.optim.AdamW(
            crit_params, lr=cfg.lr_critic,
            betas=(cfg.beta1_critic, cfg.beta2_critic),
            weight_decay=getattr(cfg, "weight_decay", 0.01),
        )

        self.generator_ema: Optional[EMA_FSDP] = None
        self.ema_weight = getattr(cfg, "ema_weight", -1.0)
        self.ema_start_step = getattr(cfg, "ema_start_step", 0)

        if target == "low_noise":
            self.high_noise_step_list = torch.tensor(cfg.high_noise_step_list, dtype=torch.long)
        else:
            self.high_noise_step_list = None

        self._build_dataloader()

        self.max_grad_norm_generator = getattr(cfg, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(cfg, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def _build_dataloader(self):
        cfg = self.config
        from mova.registry import DATASETS

        if self.i2v:
            from mova.datasets.lmdb_dataset import ShardingLMDBDataset, lmdb_collate_fn
            ds = ShardingLMDBDataset(cfg.data_path, max_pair=int(1e8))
            collate = lmdb_collate_fn
        else:
            from mova.datasets.text_prompt_dataset import text_collate_fn
            ds_cfg = dict(cfg.dataset)
            ds = DATASETS.build(ds_cfg)
            collate = text_collate_fn

        if self.world_size > 1:
            if self.mesh is not None and self._dp_size > 1:
                dp_group = self.mesh["dp"].get_group()
                dp_rank = dist.get_rank(dp_group)
                print(f"[DP DEBUG] rank={self.global_rank}, dp_size={self._dp_size}, dp_rank={dp_rank}, dp_group={dp_group}")
                sampler = torch_dist_data.DistributedSampler(
                    ds, num_replicas=self._dp_size, rank=dp_rank,
                    shuffle=True, drop_last=True,
                )
            else:
                sampler = torch_dist_data.DistributedSampler(ds, shuffle=True, drop_last=True)
            loader = DataLoader(
                ds, batch_size=cfg.batch_size, sampler=sampler,
                num_workers=getattr(cfg, "num_workers", 8),
                collate_fn=collate, pin_memory=True,
            )
        else:
            loader = DataLoader(
                ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                num_workers=getattr(cfg, "num_workers", 8),
                collate_fn=collate, pin_memory=True,
            )
        if self.is_main:
            print(f"[Distill] Building ShardingLMDBDataset: {cfg.data_path} Dataset size: {len(ds)}")
        self.dataloader = _cycle(loader)

    # ============================================================
    # Text encoding
    # ============================================================
    def _encode_text(self, text_prompts, device):
        m = self._modules
        return encode_text_prompts(
            prompts=text_prompts,
            tokenizer=m.tokenizer,
            text_encoder=m.text_encoder,
            device=device,
            dtype=self.dtype,
        )

    # ============================================================
    # I2V helpers
    # ============================================================
    def _encode_first_frame(self, img_rgb: torch.Tensor) -> torch.Tensor:
        from mova.distill.model.builder import normalize_video_latents

        m = self._modules
        B = img_rgb.shape[0]
        cfg_shape = self.config.image_or_video_shape
        num_frames = cfg_shape[1]
        H_lat, W_lat = cfg_shape[3], cfg_shape[4]
        target_H = H_lat * 8
        target_W = W_lat * 8
        C = img_rgb.shape[1]
        num_raw_frames = (num_frames - 1) * 4 + 1

        if img_rgb.shape[2] != target_H or img_rgb.shape[3] != target_W:
            img_rgb = torch.nn.functional.interpolate(
                img_rgb, size=(target_H, target_W), mode="bilinear", align_corners=False,
            )

        with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype):
            vae_input = torch.cat([
                img_rgb.unsqueeze(2),
                torch.zeros(B, C, num_raw_frames - 1, target_H, target_W,
                            device=img_rgb.device, dtype=img_rgb.dtype),
            ], dim=2)
            y_vae = m.video_vae.encode(vae_input).latent_dist.mode()
            y_vae = normalize_video_latents(m.video_vae, y_vae)

        F_lat = y_vae.shape[2]
        H_lat_actual = y_vae.shape[3]
        W_lat_actual = y_vae.shape[4]
        msk = torch.zeros(B, 4, F_lat, H_lat_actual, W_lat_actual,
                          device=y_vae.device, dtype=y_vae.dtype)
        msk[:, :, 0, :, :] = 1
        y = torch.cat([msk, y_vae], dim=1)

        self._move_module_to_device(m.video_vae, "cpu")
        return y

    # ============================================================
    # One step
    # ============================================================
    def _maybe_build_x_bound(self, batch_size, cond, y=None, audio_latents=None):
        cfg = self.config
        shape = list(cfg.image_or_video_shape)
        shape[0] = batch_size
        teacher = self.model.high_noise_model
        assert teacher is not None

        with torch.no_grad():
            noise = torch.randn(shape, device=self.device, dtype=self.dtype)
            noisy = noise
            for idx, cur_t in enumerate(self.high_noise_step_list):
                tid = (1000 - int(cur_t)) * torch.ones(
                    noise.shape[:2], device=self.device, dtype=torch.int64)
                _, denoised = teacher(
                    noisy_image_or_video=noisy, conditional_dict=cond,
                    timestep_id=tid, y=y, audio_latents=audio_latents)
                if idx != len(self.high_noise_step_list) - 1:
                    next_id = (1000 - int(self.high_noise_step_list[idx + 1])) * torch.ones(
                        noise.shape[:2], device=self.device, dtype=torch.int64)
                    noisy = self.model.scheduler.add_noise_high(
                        denoised.flatten(0, 1),
                        torch.randn_like(denoised.flatten(0, 1)),
                        next_id.flatten(0, 1),
                        teacher.timestep_bound,
                    ).unflatten(0, denoised.shape[:2])
                else:
                    noisy = denoised
            self.model.x_bound = noisy

    def fwdbwd_one_step(self, batch, train_generator: bool):
        cfg = self.config
        m = self._modules
        self.model.eval()
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        shape = list(cfg.image_or_video_shape)
        shape[0] = batch_size

        initial_latent = None
        y = None

        with torch.no_grad():
            text_offload = getattr(cfg, "text_encoder_cpu_offload", True)
            current_device = torch.device(f'cuda:{torch.cuda.current_device()}')
            if text_offload:
                self._move_module_to_device(m.text_encoder, current_device)
            cond = {"prompt_embeds": self._encode_text(text_prompts, device=current_device)}
            if not getattr(self, "_uncond_cache", None):
                neg = [cfg.negative_prompt] * batch_size
                self._uncond_cache = {
                    "prompt_embeds": self._encode_text(neg, device=current_device).detach()
                }
            uncond = self._uncond_cache
            if text_offload:
                self._move_module_to_device(m.text_encoder, "cpu")

            if self.i2v:
                image_latent = batch["ode_latent"][:, -1][:, 0:1].to(
                    device=self.device, dtype=self.dtype)
                initial_latent = image_latent
                img_rgb = batch["img"].to(device=self.device, dtype=self.dtype)
                y = self._encode_first_frame(img_rgb)

            if cfg.training_target == "low_noise":
                self._maybe_build_x_bound(batch_size, cond, y=y)

        if self._audio_dit_offload:
            self._move_module_to_device(m.audio_dit, self.device)
        if self._bridge_offload:
            self._move_module_to_device(m.dual_tower_bridge, self.device)

        print(f"[fwdbwd_one_step] initial_latent:{initial_latent.shape if initial_latent is not None else None}")
        
        if self._dp_size > 1 and self.step % 100 == 0:
            import hashlib
            data_hash = hashlib.md5(str(text_prompts[0]).encode()).hexdigest()[:8]
            print(f"[DP VERIFICATION] rank={self.global_rank}, dp_size={self._dp_size}, "
                  f"batch_size={batch_size}, first_prompt_hash={data_hash}, "
                  f"first_prompt='{text_prompts[0][:50]}...'")
        
        if train_generator:
            loss, log = self.model.generator_loss(
                image_or_video_shape=shape,
                conditional_dict=cond, unconditional_dict=uncond,
                initial_latent=initial_latent, y=y,
                cp_mesh=self.cp_mesh,
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            loss.backward()
            active = self.model.generator.video_dit
            if hasattr(active, "clip_grad_norm_"):
                grad_norm = active.clip_grad_norm_(self.max_grad_norm_generator)
            else:
                raw_norm = torch.nn.utils.clip_grad_norm_(active.parameters(), self.max_grad_norm_generator)
                grad_norm = torch.tensor(raw_norm)
            
            if self._dp_size > 1 and self.step % 100 == 0:
                first_grad = None
                for name, p in active.named_parameters():
                    if p.grad is not None:
                        first_grad = p.grad.flatten()[:5].tolist()
                        first_grad_name = name
                        break
                print(f"[DP VERIFICATION] rank={self.global_rank}, loss={loss.item():.6f}, "
                      f"grad_norm={grad_norm.item():.4f}, first_grad({first_grad_name})={first_grad}")

            if self._audio_dit_offload:
                self._move_module_to_device(m.audio_dit, "cpu")
            if self._bridge_offload:
                self._move_module_to_device(m.dual_tower_bridge, "cpu")
            print(f"[fwdbwd_one_step] generator_loss: {loss.detach()}, generator_grad_norm: {grad_norm.detach()}")
            log.update({"generator_loss": loss.detach(), "generator_grad_norm": grad_norm.detach()})
            return log
        else:
            loss, log = self.model.critic_loss(
                image_or_video_shape=shape,
                conditional_dict=cond, unconditional_dict=uncond,
                initial_latent=initial_latent, y=y,
                cp_mesh=self.cp_mesh,
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            loss.backward()
            active = self.model.fake_score.video_dit
            if hasattr(active, "clip_grad_norm_"):
                grad_norm = active.clip_grad_norm_(self.max_grad_norm_critic)
            else:
                raw_norm = torch.nn.utils.clip_grad_norm_(active.parameters(), self.max_grad_norm_critic)
                grad_norm = torch.tensor(raw_norm)

            if self._audio_dit_offload:
                self._move_module_to_device(m.audio_dit, "cpu")
            if self._bridge_offload:
                self._move_module_to_device(m.dual_tower_bridge, "cpu")

            print(f"[fwdbwd_one_step] critic_loss: {loss.detach()}, critic_grad_norm: {grad_norm.detach()}")
            log.update({"critic_loss": loss.detach(), "critic_grad_norm": grad_norm.detach()})
            return log

    # ============================================================
    # Save
    # ============================================================
    def save(self):
        gen_dit = self.model.generator.video_dit
        crit_dit = self.model.fake_score.video_dit

        gen_sd = fsdp_state_dict(gen_dit)
        crit_sd = fsdp_state_dict(crit_dit)

        state = {"generator": gen_sd, "critic": crit_sd}
        if self.generator_ema is not None and self.ema_weight > 0:
            state["generator_ema"] = self.generator_ema.state_dict()

        if self.is_main:
            ckpt_dir = os.path.join(self.config.logdir, f"checkpoint_step_{self.step:06d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(state, os.path.join(ckpt_dir, "model.pt"))
            print(f"[Save] {ckpt_dir}/model.pt")

    # ============================================================
    # Train
    # ============================================================
    def train(self):
        cfg = self.config
        max_steps = cfg.max_steps
        if self.is_main:
            print(f"[Distill] Training {cfg.training_target} for {max_steps} steps")
        start_step = self.step

        remaining_steps = max(0, start_step + max_steps - self.step)

        with tqdm(total=remaining_steps, initial=0, desc="Training", disable=not self.is_main) as pbar:
            for _ in range(remaining_steps):
                train_gen = (self.step % cfg.dfake_gen_update_ratio == 0)
                print(f"[Distill] train_gen:{train_gen} step:{self.step}")
                if train_gen:
                    self.generator_optimizer.zero_grad(set_to_none=True)
                    _ = self.fwdbwd_one_step(next(self.dataloader), True)
                    self.generator_optimizer.step()
                    if self.generator_ema is None and self.ema_weight > 0 and self.step >= self.ema_start_step:
                        active = self.model.generator.video_dit
                        self.generator_ema = (EMA_FSDP if self.world_size > 1 else EMA)(active, decay=self.ema_weight)
                    elif self.generator_ema is not None:
                        active = self.model.generator.video_dit
                        self.generator_ema.update(active)

                self.critic_optimizer.zero_grad(set_to_none=True)
                _ = self.fwdbwd_one_step(next(self.dataloader), False)
                self.critic_optimizer.step()

                self.step += 1

                if (not getattr(cfg, "no_save", False)) and self.step % cfg.log_iters == 0:
                    self.save()

                if self.step % getattr(cfg, "gc_interval", 100) == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

                if self.is_main:
                    now = time.time()
                    if self.previous_time is not None:
                        iter_time = now - self.previous_time
                        pbar.set_postfix({"iter": f"{iter_time:.2f}s", "step": self.step})
                    self.previous_time = now
                    pbar.update(1)

        self.save()
        if self.is_main:
            print(f"[Distill] {cfg.training_target} stage complete.")