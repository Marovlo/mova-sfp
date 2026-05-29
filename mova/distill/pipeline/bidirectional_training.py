"""
Bidirectional N-step student sampler used during DMD training to backward-simulate
the generator's noisy input. Adapted from Self-Forcing-Plus
`pipeline/bidirectional_training.py` to use the MOVA-side scheduler & wrappers.
"""

from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist


class BidirectionalTrainingPipeline(torch.nn.Module):
    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler,
        generator,                    # MOVAVideoDiTWrapper
        boundary_step: int,
        training_target: str,
        timestep_shift: float = 5.0,
    ):
        super().__init__()
        self.training_target = training_target
        self.scheduler = scheduler
        self.generator = generator

        self.denoising_step_list = denoising_step_list
        if isinstance(self.denoising_step_list, list):
            self.denoising_step_list = torch.tensor(self.denoising_step_list, dtype=torch.long)
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        self.boundary_step = boundary_step
        self.timestep_shift = timestep_shift
        self.timestep_bound = torch.tensor([boundary_step], dtype=torch.float64)
        if timestep_shift > 1:
            self.timestep_bound = (
                timestep_shift
                * (self.timestep_bound / 1000.0)
                / (1 + (timestep_shift - 1) * (self.timestep_bound / 1000.0))
                * 1000.0
            )

    def _broadcast_exit_index(self, num_steps: int, device) -> int:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            idx = torch.randint(0, num_steps, (1,), device=device)
        else:
            idx = torch.empty(1, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(idx, src=0)
        return int(idx.item())

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        y=None,
        audio_latents=None,
        cp_mesh=None,
        **conditional_dict,
    ):
        """Run an N-step trajectory of the student. Stops at a uniformly random
        timestep so that backward simulation produces a valid noisy input for
        DMD's score loss (see SFP DMD2 sec 4.5).
        """
        noisy = noise
        num_steps = len(self.denoising_step_list)
        exit_idx = self._broadcast_exit_index(num_steps, device=noise.device)

        for index, current_timestep in enumerate(self.denoising_step_list):
            print(f"[inference_with_trajectory] index:{index} current_timestep:{current_timestep}")
            timestep_id = (1000 - int(current_timestep)) * torch.ones(
                noise.shape[:2], device=noise.device, dtype=torch.int64,
            )
            if index != exit_idx:
                with torch.no_grad():
                    flow_pred, denoised_pred = self.generator(
                        noisy_image_or_video=noisy,
                        conditional_dict=conditional_dict,
                        timestep_id=timestep_id,
                        y=y,
                        audio_latents=audio_latents,
                        cp_mesh=cp_mesh,
                    )
                    next_timestep_id = (1000 - int(self.denoising_step_list[index + 1])) * torch.ones(
                        noise.shape[:2], dtype=torch.long, device=noise.device,
                    )
                    if self.training_target == "high_noise":
                        noisy = self.scheduler.add_noise_high(
                            denoised_pred.flatten(0, 1),
                            noise.flatten(0, 1),
                            next_timestep_id.flatten(0, 1),
                            self.timestep_bound,
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        noisy = self.scheduler.add_noise_low(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep_id.flatten(0, 1),
                            self.timestep_bound,
                        ).unflatten(0, denoised_pred.shape[:2])
                    del flow_pred, denoised_pred
                    torch.cuda.empty_cache()
            else:
                from mova.distill.model.dmd import log_cuda_memory_simple
                log_cuda_memory_simple(f'before generator call (index={index})')
                flow_pred, denoised_pred = self.generator(
                    noisy_image_or_video=noisy,
                    conditional_dict=conditional_dict,
                    timestep_id=timestep_id,
                    y=y,
                    audio_latents=audio_latents,
                    cp_mesh=cp_mesh,
                )
                log_cuda_memory_simple(f'after generator call (index={index})')
                break

        return flow_pred, denoised_pred
