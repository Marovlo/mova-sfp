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
import logging
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, distributed as torch_dist_data

from mova.diffusion.models.wan_video_dit import DiTBlock as WanDiTBlock
from mova.distill.model.builder import build_distill_modules
from mova.distill.model.dmd import MOVADMD
from mova.distill.model.mova_dit_wrapper import MOVAVideoDiTWrapper
from mova.distill.utils.distributed import (
    EMA_FSDP, fsdp_state_dict, fsdp_wrap, launch_distributed_job,
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

    def __init__(self, config):
        self.config = config
        self.step = 0

        # ---------- distributed ----------
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

        # ---------- build distill modules ----------
        if self.is_main:
            print(f"[Distill] stage={config.training_target}, i2v={self.i2v}")
            print(f"[Distill] Building modules from {config.pretrained_path}")

        self._modules = build_distill_modules(
            pretrained_path=config.pretrained_path,
            use_gradient_checkpointing=config.gradient_checkpointing,
            use_gradient_checkpointing_offload=getattr(config, "gradient_checkpointing_offload", False),
            torch_dtype=self.dtype,
        )
        self.master = self._modules.master

        self._setup()

    # ============================================================
    # Setup
    # ============================================================
    def _setup(self):
        cfg = self.config
        target = cfg.training_target  # "high_noise" or "low_noise"
        m = self._modules

        # -- high_noise_teacher for low stage --
        high_noise_teacher = None
        if target == "low_noise":
            import copy
            distill_path = cfg.high_noise_distill_ckpt  # .pt from stage 1
            if self.is_main:
                print(f"[Distill] Loading high-noise distilled ckpt: {distill_path}")
            teacher_dit_high = copy.deepcopy(m.real_high)
            state = torch.load(distill_path, map_location="cpu")
            if "generator" in state:
                state = state["generator"]
            teacher_dit_high.load_state_dict(state, strict=True)
            teacher_dit_high.requires_grad_(False).eval()

            high_noise_teacher = MOVAVideoDiTWrapper(
                master_pipeline=self.master,
                video_dit_high=teacher_dit_high,
                video_dit_low=m.real_low,
                target="high_noise",
                boundary_step=cfg.boundary_step,
                timestep_shift=cfg.timestep_shift,
            )

        # -- wrappers --
        def _mk(dit_h, dit_l, tgt):
            return MOVAVideoDiTWrapper(
                master_pipeline=self.master,
                video_dit_high=dit_h, video_dit_low=dit_l,
                target=tgt, boundary_step=cfg.boundary_step,
                timestep_shift=cfg.timestep_shift,
            )

        generator = _mk(m.generator_high, m.generator_low, target)
        real_score = _mk(m.real_high, m.real_low, target)
        fake_score = _mk(m.fake_high, m.fake_low, target)
        for p in real_score.parameters():
            p.requires_grad_(False)

        # -- FSDP wrap --
        wrap_kw = dict(
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            wrap_strategy=cfg.fsdp_wrap_strategy,
            transformer_module={WanDiTBlock} if cfg.fsdp_wrap_strategy == "transformer" else None,
        )
        for wrapper in (generator, real_score, fake_score):
            wrapper.video_dit_high = fsdp_wrap(wrapper.video_dit_high, **wrap_kw)
            wrapper.video_dit_low = fsdp_wrap(wrapper.video_dit_low, **wrap_kw)
            wrapper.model = wrapper.video_dit_high if target == "high_noise" else wrapper.video_dit_low

        if high_noise_teacher is not None:
            high_noise_teacher.video_dit_high = fsdp_wrap(
                high_noise_teacher.video_dit_high, cpu_offload=False, **wrap_kw,
            )

        # -- peripherals placement --
        text_offload = getattr(cfg, "text_encoder_cpu_offload", True)
        self.master.text_encoder.to("cpu" if text_offload else self.device)
        self.master.text_encoder.requires_grad_(False)
        if self.i2v:
            # video_vae needed on-GPU for first-frame encoding
            self.master.video_vae.to(self.device, dtype=self.dtype)
        else:
            self.master.video_vae.to("cpu")
        self.master.video_vae.requires_grad_(False)
        self.master.audio_vae.to("cpu")
        self.master.audio_vae.requires_grad_(False)
        self.master.audio_dit.to(self.device, dtype=self.dtype)
        self.master.audio_dit.requires_grad_(False)
        self.master.dual_tower_bridge.to(self.device, dtype=self.dtype)
        self.master.dual_tower_bridge.requires_grad_(False)

        # -- DMD model --
        self.model = MOVADMD(
            config=cfg, device=self.device,
            generator=generator, real_score=real_score, fake_score=fake_score,
            master_pipeline=self.master,
            high_noise_teacher=high_noise_teacher,
        )

        # -- optimizers --
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

        # -- EMA --
        self.generator_ema: Optional[EMA_FSDP] = None
        self.ema_weight = getattr(cfg, "ema_weight", -1.0)
        self.ema_start_step = getattr(cfg, "ema_start_step", 0)

        # -- high_noise_step_list (for low-noise x_bound build) --
        if target == "low_noise":
            self.high_noise_step_list = torch.tensor(cfg.high_noise_step_list, dtype=torch.long)
        else:
            self.high_noise_step_list = None

        # -- dataloader --
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

        sampler = torch_dist_data.DistributedSampler(ds, shuffle=True, drop_last=True)
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, sampler=sampler,
            num_workers=getattr(cfg, "num_workers", 8),
            collate_fn=collate, pin_memory=True,
        )
        if self.is_main:
            print(f"[Distill] Dataset size: {len(ds)}")
        self.dataloader = _cycle(loader)

    # ============================================================
    # I2V helpers
    # ============================================================
    def _encode_first_frame(self, img_rgb: torch.Tensor) -> torch.Tensor:
        """Encode first-frame RGB tensor to the MOVA 20-channel y condition.
        img_rgb: [B, C, H, W] in [-1, 1] float.
        Returns y: [B, 20, F_lat, H_lat, W_lat].
        """
        pipe = self.master
        B = img_rgb.shape[0]
        cfg_shape = self.config.image_or_video_shape  # [B, F_lat, C, H_lat, W_lat]
        num_frames = cfg_shape[1]  # F_lat (e.g. 21)
        H_lat, W_lat = cfg_shape[3], cfg_shape[4]
        # Target pixel resolution = latent × vae_stride (8)
        target_H = H_lat * 8
        target_W = W_lat * 8
        C = img_rgb.shape[1]
        num_raw_frames = (num_frames - 1) * 4 + 1  # 81 for F_lat=21

        # Resize img to target resolution if needed
        if img_rgb.shape[2] != target_H or img_rgb.shape[3] != target_W:
            img_rgb = torch.nn.functional.interpolate(
                img_rgb, size=(target_H, target_W), mode="bilinear", align_corners=False,
            )

        with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype):
            vae_input = torch.cat([
                img_rgb.unsqueeze(2),  # [B, C, 1, H, W]
                torch.zeros(B, C, num_raw_frames - 1, target_H, target_W,
                            device=img_rgb.device, dtype=img_rgb.dtype),
            ], dim=2)
            y_vae = pipe.video_vae.encode(vae_input).latent_dist.mode()
            y_vae = pipe.normalize_video_latents(y_vae)

        # Build mask: [B, 4, F_lat, H_lat, W_lat]
        F_lat = y_vae.shape[2]
        H_lat_actual = y_vae.shape[3]
        W_lat_actual = y_vae.shape[4]
        msk = torch.zeros(B, 4, F_lat, H_lat_actual, W_lat_actual,
                          device=y_vae.device, dtype=y_vae.dtype)
        msk[:, :, 0, :, :] = 1
        y = torch.cat([msk, y_vae], dim=1)  # [B, 20, F_lat, H_lat, W_lat]
        return y

    # ============================================================
    # One step
    # ============================================================
    def _maybe_build_x_bound(self, batch_size, cond, y=None, audio_latents=None):
        """Low-noise stage: run frozen high_noise teacher from pure noise → x_bound."""
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
        self.model.eval()
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        shape = list(cfg.image_or_video_shape)
        shape[0] = batch_size

        # -- I2V data --
        initial_latent = None
        y = None

        with torch.no_grad():
            # text encoder
            text_offload = getattr(cfg, "text_encoder_cpu_offload", True)
            if text_offload:
                self.master.text_encoder.to(self.device)
            cond = {"prompt_embeds": self.master._get_t5_prompt_embeds(text_prompts, device=self.device)}
            if not getattr(self, "_uncond_cache", None):
                neg = [cfg.negative_prompt] * batch_size
                self._uncond_cache = {
                    "prompt_embeds": self.master._get_t5_prompt_embeds(neg, device=self.device).detach()
                }
            uncond = self._uncond_cache
            if text_offload:
                self.master.text_encoder.to("cpu")

            if self.i2v:
                # batch from ShardingLMDBDataset:
                #   ode_latent: [B, 1, 21, 16, H_lat, W_lat] (last frame = clean first-frame latent)
                #   img: [B, C, H, W]  (first frame RGB in [-1,1])
                image_latent = batch["ode_latent"][:, -1][:, 0:1].to(
                    device=self.device, dtype=self.dtype)
                initial_latent = image_latent
                img_rgb = batch["img"].to(device=self.device, dtype=self.dtype)
                y = self._encode_first_frame(img_rgb)

            if cfg.training_target == "low_noise":
                self._maybe_build_x_bound(batch_size, cond, y=y)

        if train_generator:
            loss, log = self.model.generator_loss(
                image_or_video_shape=shape,
                conditional_dict=cond, unconditional_dict=uncond,
                initial_latent=initial_latent, y=y,
            )
            torch.cuda.empty_cache()
            loss.backward()
            active = self.model.generator.video_dit_high if cfg.training_target == "high_noise" \
                else self.model.generator.video_dit_low
            grad_norm = active.clip_grad_norm_(self.max_grad_norm_generator)
            log.update({"generator_loss": loss.detach(), "generator_grad_norm": grad_norm.detach()})
            return log

        loss, log = self.model.critic_loss(
            image_or_video_shape=shape,
            conditional_dict=cond, unconditional_dict=uncond,
            initial_latent=initial_latent, y=y,
        )
        loss.backward()
        active = self.model.fake_score.video_dit_high if cfg.training_target == "high_noise" \
            else self.model.fake_score.video_dit_low
        grad_norm = active.clip_grad_norm_(self.max_grad_norm_critic)
        log.update({"critic_loss": loss.detach(), "critic_grad_norm": grad_norm.detach()})
        return log

    # ============================================================
    # Save
    # ============================================================
    def save(self):
        cfg = self.config
        is_high = cfg.training_target == "high_noise"
        gen_dit = self.model.generator.video_dit_high if is_high else self.model.generator.video_dit_low
        crit_dit = self.model.fake_score.video_dit_high if is_high else self.model.fake_score.video_dit_low

        gen_sd = fsdp_state_dict(gen_dit)
        crit_sd = fsdp_state_dict(crit_dit)

        state = {"generator": gen_sd, "critic": crit_sd}
        if self.generator_ema is not None and self.ema_weight > 0:
            state["generator_ema"] = self.generator_ema.state_dict()

        if self.is_main:
            ckpt_dir = os.path.join(cfg.logdir, f"checkpoint_step_{self.step:06d}")
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

        while self.step < start_step + max_steps:
            train_gen = (self.step % cfg.dfake_gen_update_ratio == 0)

            if train_gen:
                self.generator_optimizer.zero_grad(set_to_none=True)
                _ = self.fwdbwd_one_step(next(self.dataloader), True)
                self.generator_optimizer.step()
                # EMA
                if self.generator_ema is None and self.ema_weight > 0 and self.step >= self.ema_start_step:
                    active = self.model.generator.video_dit_high if cfg.training_target == "high_noise" \
                        else self.model.generator.video_dit_low
                    self.generator_ema = EMA_FSDP(active, decay=self.ema_weight)
                elif self.generator_ema is not None:
                    active = self.model.generator.video_dit_high if cfg.training_target == "high_noise" \
                        else self.model.generator.video_dit_low
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
                    print(f"[step {self.step}] iter={now - self.previous_time:.2f}s")
                self.previous_time = now

        self.save()
        if self.is_main:
            print(f"[Distill] {cfg.training_target} stage complete.")
