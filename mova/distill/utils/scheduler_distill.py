"""
MoE-aware FlowMatchScheduler for MOVA DMD distillation.

This file adds the `add_noise_high` / `add_noise_low` corruption operators
(borrowed verbatim in spirit from Self-Forcing-Plus, with sigma lookup adapted to
the MOVA scheduler convention).

The scheduler instance still relies on the *training* `train_sigmas` / `train_timesteps`
table (1000 steps), regardless of whether `set_timesteps` was later called for
inference: we re-derive a fixed table to make timestep-id lookups deterministic.
"""

from __future__ import annotations

import torch

from mova.diffusion.schedulers.flow_match import FlowMatchScheduler
from mova.registry import DIFFUSION_SCHEDULERS


def _shifted_sigmas(num_train_timesteps: int, shift: float, device) -> torch.Tensor:
    """Reproduce the SFP/Wan training-time sigma grid (1000 entries, sigma_min=0,
    extra_one_step=True), independent from any inference-time set_timesteps call.
    """
    sigma = torch.linspace(1.0, 0.0, num_train_timesteps + 1, device=device)[:-1]
    sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
    return sigma


@DIFFUSION_SCHEDULERS.register_module()
class FlowMatchSchedulerDistill(FlowMatchScheduler):
    """A FlowMatchScheduler that exposes MoE-aware corruption helpers used by
    Self-Forcing-Plus's DMD trainer (high/low noise variants).

    All semantics match SFP `utils/scheduler.py:FlowMatchScheduler` once the
    `_train_sigmas` lookup table is configured to the same shift.
    """

    def __init__(self, *args, num_train_timesteps: int = 1000, shift: float = 5.0, **kwargs):
        kwargs.setdefault("num_train_timesteps", num_train_timesteps)
        kwargs.setdefault("shift", shift)
        kwargs.setdefault("sigma_min", 0.0)
        kwargs.setdefault("extra_one_step", True)
        super().__init__(*args, **kwargs)
        # Persistent training sigma table (used for add_noise / id lookup)
        self._train_sigmas = _shifted_sigmas(num_train_timesteps, shift, device=torch.device("cpu"))
        self._train_timesteps = self._train_sigmas * num_train_timesteps

    # ---------- helpers ----------
    def _to(self, t: torch.Tensor, device) -> torch.Tensor:
        return t.to(device)

    def get_train_sigmas(self, device) -> torch.Tensor:
        return self._train_sigmas.to(device)

    def get_train_timesteps(self, device) -> torch.Tensor:
        return self._train_timesteps.to(device)

    @staticmethod
    def calculate_alpha_beta_high(sigma: torch.Tensor, sigma_bound: torch.Tensor):
        alpha = (1 - sigma) / (1 - sigma_bound)
        beta = torch.sqrt(torch.clamp(sigma**2 - (alpha * sigma_bound) ** 2, min=0.0))
        return alpha, beta

    @staticmethod
    def calculate_alpha_beta_low(sigma: torch.Tensor, sigma_bound: torch.Tensor):
        beta = sigma / sigma_bound
        alpha = 1 - beta
        return alpha, beta

    def shift_boundary_step(self, boundary_step: int) -> torch.Tensor:
        """Map boundary step (in 0-1000 raw units) through the sigma shift,
        returning the shifted boundary step on the same device-less scale.
        """
        x = torch.tensor([boundary_step], dtype=torch.float64)
        if self.shift > 1:
            x = self.shift * (x / self.num_train_timesteps) / (
                1 + (self.shift - 1) * (x / self.num_train_timesteps)
            ) * self.num_train_timesteps
        return x

    # ---------- forward corruption ----------
    def add_noise(self, original_samples, noise, timestep_id):
        """Override the base-class scalar implementation to support batched
        timestep_id tensors of shape [B*T] (the SFP convention).
        """
        sigmas = self.get_train_sigmas(noise.device)
        sigma = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)

    def add_noise_high(self, original_samples, noise, timestep_id, timestep_bound):
        sigmas = self.get_train_sigmas(noise.device)
        timesteps_table = self.get_train_timesteps(noise.device)
        timestep_bound = timestep_bound.to(timesteps_table.device)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        timestep_id_bound = torch.argmin(
            (timesteps_table.unsqueeze(0) - timestep_bound.unsqueeze(1)).abs(), dim=1
        )
        sigma_t_bound = sigmas[timestep_id_bound].reshape(-1, 1, 1, 1)
        alpha, beta = self.calculate_alpha_beta_high(sigma_t, sigma_t_bound)
        return (alpha * original_samples + beta * noise).type_as(noise)

    def add_noise_low(self, original_samples, noise, timestep_id, timestep_bound):
        # SFP add_noise_low ignores timestep_bound and behaves identically to add_noise.
        return self.add_noise(original_samples, noise, timestep_id)

    # ---------- back-compat property used by SFP-style code ----------
    @property
    def sigmas_train(self) -> torch.Tensor:  # alias for clarity
        return self._train_sigmas
