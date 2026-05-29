"""
DMD (Distribution Matching Distillation) wrapper for MOVA.

Uses MOVAVideoDiTWrapper for generator / real_score / fake_score.
No centralized master pipeline — each wrapper owns its inference, and shared
resources (audio_dit, dual_tower_bridge) are managed directly for offload.

Adapted I/O shapes to MOVA's [B, C, F, H, W] convention internally; SFP-style
[B, F, C, H, W] is preserved at the public interface so the trainer code is
unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from mova.distill.pipeline.bidirectional_training import BidirectionalTrainingPipeline


def _move_module_to_device(module, device):
    module.to(device)
    with torch.no_grad():
        for p in module.parameters():
            p.data = p.data.to(device=device)
        for b in module.buffers():
            b.data = b.data.to(device=device)

def log_cuda_memory_simple(tag: str):
    """
    简单的 CUDA 显存追踪，只打印关键信息。
    """
    if not torch.cuda.is_available():
        return
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"[CUDA {tag}] alloc={allocated:.2f}GB | reserved={reserved:.2f}GB | max_alloc={max_allocated:.2f}GB")


class MOVADMD(torch.nn.Module):
    """Self-contained DMD module: generator + real_score + fake_score wrappers.

    No centralized master pipeline — shared resources are managed independently.
    Each wrapper owns its video DiT and self-contained inference logic.
    """

    def __init__(
        self,
        config,
        device,
        generator,
        real_score,
        fake_score,
        scheduler,
        shared_audio_dit=None,
        shared_dual_tower_bridge=None,
        high_noise_teacher=None,
    ):
        super().__init__()
        self.config = config
        self.device = device

        self.generator = generator
        self.real_score = real_score
        self.fake_score = fake_score
        self.high_noise_model = high_noise_teacher

        self.shared_audio_dit = shared_audio_dit
        self.shared_dual_tower_bridge = shared_dual_tower_bridge
        self.scheduler = scheduler

        self.training_target = config.training_target
        self.boundary_step = config.boundary_step
        self.timestep_shift = getattr(config, "timestep_shift", 5.0)
        self.dtype = torch.bfloat16 if getattr(config, "mixed_precision", True) else torch.float32

        self.num_train_timestep = getattr(config, "num_train_timestep", 1000)
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)

        if self.training_target == "high_noise":
            moe_train_step = self.num_train_timestep - self.boundary_step
            self.min_timestep = int(self.boundary_step + moe_train_step * 0.04)
            self.max_timestep = int(self.boundary_step + moe_train_step * 0.96)
        else:
            moe_train_step = self.boundary_step
            self.min_timestep = int(moe_train_step * 0.04)
            self.max_timestep = int(moe_train_step * 0.96)

        self.real_guidance_scale = getattr(config, "guidance_scale", 3.0)
        self.fake_guidance_scale = 0.0

        self.timestep_bound = torch.tensor([self.boundary_step], dtype=torch.float64)
        if self.timestep_shift > 1:
            self.timestep_bound = (
                self.timestep_shift
                * (self.timestep_bound / self.num_train_timestep)
                / (1 + (self.timestep_shift - 1) * (self.timestep_bound / self.num_train_timestep))
                * self.num_train_timestep
            )
        self.sigma_bound = self.timestep_bound / self.num_train_timestep

        self.inference_pipeline: Optional[BidirectionalTrainingPipeline] = None
        self.num_training_frames = getattr(config, "num_training_frames", 21)
        self.num_frame_per_block = getattr(config, "num_frame_per_block", 3)
        self.same_step_across_blocks = getattr(config, "same_step_across_blocks", True)
        self.independent_first_frame = getattr(config, "independent_first_frame", False)

        self.x_bound: Optional[torch.Tensor] = None

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _get_timestep(self, lo, hi, batch_size, num_frame, num_frame_per_block, uniform_timestep=True):
        if uniform_timestep:
            t = torch.randint(lo, hi, [batch_size, 1], device=self.device, dtype=torch.long)
            return t.repeat(1, num_frame)
        t = torch.randint(lo, hi, [batch_size, num_frame], device=self.device, dtype=torch.long)
        t = t.reshape(batch_size, -1, num_frame_per_block)
        t[:, :, 1:] = t[:, :, 0:1]
        return t.reshape(batch_size, -1)

    def _initialize_inference_pipeline(self):
        if self.inference_pipeline is None:
            self.inference_pipeline = BidirectionalTrainingPipeline(
                denoising_step_list=self.config.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                boundary_step=self.boundary_step,
                training_target=self.training_target,
                timestep_shift=self.timestep_shift,
            )


    def _run_generator(self, image_or_video_shape, conditional_dict, audio_latents=None,
                       initial_latent=None, y=None, cp_mesh=None):
        self._initialize_inference_pipeline()

        i2v = getattr(self.config, "i2v", False)

        if i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1,
                           *image_or_video_shape[2:]]
        else:
            noise_shape = list(image_or_video_shape)

        if self.training_target == "low_noise":
            assert self.x_bound is not None, "low_noise stage requires x_bound to be set"
            noise = self.x_bound.to(device=self.device, dtype=self.dtype)
        else:
            noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype)

        if initial_latent is not None:
            conditional_dict = dict(conditional_dict)
            conditional_dict["initial_latent"] = initial_latent

        flow_pred, pred_image = self.inference_pipeline.inference_with_trajectory(
            noise=noise, y=y, audio_latents=audio_latents, cp_mesh=cp_mesh, **conditional_dict,
        )

        torch.cuda.empty_cache()
        return flow_pred, pred_image, None, noise

    def _compute_kl_grad(self, noisy, clean, timestep_id, conditional_dict, unconditional_dict,
                          normalization=True, y=None, audio_latents=None, cp_mesh=None):
                     
        log_cuda_memory_simple('before pred_real_cond')
        _, pred_real_cond = self.real_score(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep_id=timestep_id, y=y,
            audio_latents=audio_latents, cp_mesh=cp_mesh,
        )
        pred_real_cond = pred_real_cond.detach().cpu()
        torch.cuda.empty_cache()
        log_cuda_memory_simple('after pred_real_cond')

        log_cuda_memory_simple('before pred_real_uncond')
        _, pred_real_uncond = self.real_score(
            noisy_image_or_video=noisy,
            conditional_dict=unconditional_dict,
            timestep_id=timestep_id, y=y,
            audio_latents=audio_latents, cp_mesh=cp_mesh,
        )
        pred_real_uncond = pred_real_uncond.detach().cpu()
        torch.cuda.empty_cache()

        log_cuda_memory_simple('before pred_fake')
        _, pred_fake = self.fake_score(
            noisy_image_or_video=noisy,
            conditional_dict=conditional_dict,
            timestep_id=timestep_id, y=y,
            audio_latents=audio_latents, cp_mesh=cp_mesh,
        )
        pred_fake = pred_fake.detach().cpu()
        torch.cuda.empty_cache()
        log_cuda_memory_simple('after pred_fake')
        
        pred_real_cond = pred_real_cond.to(device=pred_real_uncond.device, dtype=pred_real_uncond.dtype)
        pred_real = pred_real_cond + (pred_real_cond - pred_real_uncond) * self.real_guidance_scale

        pred_fake = pred_fake.to(device=pred_real.device, dtype=pred_real.dtype)
        grad = pred_fake - pred_real
        clean = clean.to(device=grad.device, dtype=grad.dtype)
        if normalization:
            normalizer = torch.abs(clean - pred_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / (normalizer + 1e-8)
        grad = torch.nan_to_num(grad)
        return grad, {"dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
                      "timestep": timestep_id.detach()}

    # ------------------------------------------------------------
    # Public API: generator_loss / critic_loss
    # ------------------------------------------------------------
    def generator_loss(self, image_or_video_shape, conditional_dict, unconditional_dict,
                       clean_latent=None, initial_latent=None, y=None,
                       audio_latents=None, cp_mesh=None):

        log_cuda_memory_simple('generator_loss START')
        flow_pred, pred_image, _gradient_mask, _noise = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            audio_latents=audio_latents,
            initial_latent=initial_latent,
            y=y,
            cp_mesh=cp_mesh,
        )
        del flow_pred, _gradient_mask, _noise
        torch.cuda.empty_cache()
        log_cuda_memory_simple('after _run_generator')

        if initial_latent is not None:
            conditional_dict = dict(conditional_dict)
            conditional_dict["initial_latent"] = initial_latent
            unconditional_dict = dict(unconditional_dict)
            unconditional_dict["initial_latent"] = initial_latent

        with torch.no_grad():
            bsz, num_frame = pred_image.shape[:2]
            pred_image_detach = pred_image.detach().cpu()
            t = self._get_timestep(self.min_timestep, self.max_timestep, bsz, num_frame,
                                   self.num_frame_per_block, uniform_timestep=True)
            t = t.clamp(self.min_step, self.max_step)
            timestep_id = 1000 - t
            noise = torch.randn(bsz, num_frame, *image_or_video_shape[2:],
                                device=self.device, dtype=self.dtype)

            if self.training_target == "high_noise":
                noisy = self.scheduler.add_noise_high(
                    pred_image_detach.to(device=self.device, dtype=self.dtype).flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep_id.flatten(0, 1), self.timestep_bound,
                ).detach().unflatten(0, (bsz, num_frame))
            else:
                noisy = self.scheduler.add_noise_low(
                    pred_image_detach.to(device=self.device, dtype=self.dtype).flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep_id.flatten(0, 1), self.timestep_bound,
                ).detach().unflatten(0, (bsz, num_frame))
            
            del noise, pred_image_detach
            torch.cuda.empty_cache()
            log_cuda_memory_simple('before _compute_kl_grad')

            grad, log_dict = self._compute_kl_grad(
                noisy=noisy, clean=pred_image, timestep_id=timestep_id,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                y=y, audio_latents=audio_latents, cp_mesh=cp_mesh,
            )
            del noisy
            torch.cuda.empty_cache()

        dmd_loss = 0.5 * F.mse_loss(
            pred_image.double(),
            (pred_image.double() - grad.double()).detach(),
            reduction="mean",
        )
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"[DMD] dmd_loss: {dmd_loss.item():.6f}")
        return dmd_loss, log_dict

    def critic_loss(self, image_or_video_shape, conditional_dict, unconditional_dict,
                    clean_latent=None, initial_latent=None, y=None,
                    audio_latents=None, cp_mesh=None):
        with torch.no_grad():
            _, generated, _, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                audio_latents=audio_latents,
                initial_latent=initial_latent,
                y=y, cp_mesh=cp_mesh, 
            )

        bsz, num_frame = generated.shape[:2]
        critic_t = self._get_timestep(self.min_timestep, self.max_timestep, bsz, num_frame,
                                      self.num_frame_per_block, uniform_timestep=True)
        critic_t = critic_t.clamp(self.min_step, self.max_step)
        critic_id = 1000 - critic_t
        critic_noise = torch.randn_like(generated)

        if self.training_target == "high_noise":
            noisy_gen = self.scheduler.add_noise_high(
                generated.flatten(0, 1), critic_noise.flatten(0, 1),
                critic_id.flatten(0, 1), self.timestep_bound,
            ).unflatten(0, (bsz, num_frame))
        else:
            noisy_gen = self.scheduler.add_noise_low(
                generated.flatten(0, 1), critic_noise.flatten(0, 1),
                critic_id.flatten(0, 1), self.timestep_bound,
            ).unflatten(0, (bsz, num_frame))

        generated_cpu = generated.cpu()
        del generated
        torch.cuda.empty_cache()

        if initial_latent is not None:
            conditional_dict = dict(conditional_dict)
            conditional_dict["initial_latent"] = initial_latent

        flow_pred_fake, _ = self.fake_score(
            noisy_image_or_video=noisy_gen,
            conditional_dict=conditional_dict,
            timestep_id=critic_id, y=y,
            audio_latents=audio_latents, cp_mesh=cp_mesh,
        )
        torch.cuda.empty_cache()

        sigmas = self.scheduler.get_train_sigmas(noisy_gen.device)
        t_vals = sigmas[critic_id].reshape(-1, 1, 1, 1).to(noisy_gen.dtype)
        t_vals = t_vals.reshape(bsz * num_frame, 1, 1, 1)

        if self.training_target == "high_noise":
            s = self.sigma_bound.to(noisy_gen.device).to(noisy_gen.dtype)
            alpha, beta = self.scheduler.calculate_alpha_beta_high(t_vals, s)
            num = ((1 - s) * (t_vals - beta * beta)) * noisy_gen.flatten(0, 1) \
                  - ((1 - s) * (1 - t_vals) * beta * beta) * flow_pred_fake.flatten(0, 1)
            den = ((1 - t_vals) * beta * beta) + ((1 - s) * (t_vals - beta * beta)) * alpha
            fake_image = (num / (den + 1e-8)).unflatten(0, (bsz, num_frame))
        else:
            fake_image = (noisy_gen.flatten(0, 1) - flow_pred_fake.flatten(0, 1) * t_vals).unflatten(0, (bsz, num_frame))

        generated = generated_cpu.to(device=fake_image.device, dtype=fake_image.dtype)
        denoising_loss = torch.mean((fake_image - generated) ** 2)
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"[DMD] denoising_loss: {denoising_loss.item():.6f}")
        return denoising_loss, {"critic_timestep": critic_id.detach()}