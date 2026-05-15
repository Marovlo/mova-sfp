"""
MOVA Video-DiT wrapper for Self-Forcing-Plus style DMD distillation.

A wrapper presents one role (generator / real_score / fake_score) as a self-
contained module with the SFP `WanDiffusionWrapper` interface
`forward(noisy_image_or_video, conditional_dict, timestep_id, ...) -> (flow_pred, x_pred)`.

It owns its OWN `video_dit` and `video_dit_2` instances and, on every forward,
temporarily swaps them into the SHARED `MOVATrain` master pipeline so that
`inference_single_step` runs with the correct DiT while still using the master's
`audio_dit`, `dual_tower_bridge`, `video_vae`, `text_encoder`, etc.

Concurrency caveat: forward is *not* re-entrant across roles; the trainer calls
each role serially (matches SFP's `fwdbwd_one_step` order).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from mova.diffusion.pipelines.mova_train import MOVATrain
from mova.distill.model.builder import swap_video_dit


class _SharedRefs:
    """Plain-Python (non-nn.Module) holder so FSDP / .parameters() does not
    descend into the master pipeline."""
    __slots__ = ("pipeline",)

    def __init__(self, pipeline: MOVATrain):
        self.pipeline = pipeline


class MOVAVideoDiTWrapper(nn.Module):
    """SFP-style wrapper around a (high-noise + low-noise) pair of video DiTs
    that runs through MOVA's full dual-tower forward.

    Args:
        master_pipeline: shared `MOVATrain` whose audio_dit / bridge / vaes / etc.
                         we reuse on every forward call.
        video_dit_high : the role's high-noise DiT (active when target=="high_noise").
        video_dit_low  : the role's low-noise DiT  (active when target=="low_noise").
        target         : "high_noise" or "low_noise".
        boundary_step  : raw boundary step (0..1000) used to compute x_bound.
        timestep_shift : sigma shift used by the FlowMatchScheduler.
    """

    def __init__(
        self,
        master_pipeline: MOVATrain,
        video_dit_high: nn.Module,
        video_dit_low: nn.Module,
        target: str = "high_noise",
        boundary_step: int = 900,
        timestep_shift: float = 5.0,
    ):
        super().__init__()
        assert target in {"high_noise", "low_noise"}

        self._refs = _SharedRefs(master_pipeline)
        self.target = target
        self.timestep_shift = timestep_shift

        # Own both video DiTs as our submodules. The trainer decides which
        # roles' wrappers see `requires_grad=True` (real_score wrappers freeze
        # both, generator/fake_score wrappers train both).
        self.add_module("video_dit_high", video_dit_high)
        self.add_module("video_dit_low", video_dit_low)

        # `.model` alias keeps SFP-style trainer code that does
        # `self.model.generator.parameters()` happy.
        self.model = self.video_dit_high if target == "high_noise" else self.video_dit_low

        self.timestep_bound = torch.tensor([boundary_step], dtype=torch.float64)
        if timestep_shift > 1:
            self.timestep_bound = (
                timestep_shift
                * (self.timestep_bound / 1000.0)
                / (1 + (timestep_shift - 1) * (self.timestep_bound / 1000.0))
                * 1000.0
            )

        self.scheduler = master_pipeline.scheduler

    @property
    def pipeline(self) -> MOVATrain:
        return self._refs.pipeline

    def enable_gradient_checkpointing(self) -> None:
        # MOVA reads gradient checkpointing flags off the pipeline; nothing to do.
        return

    # ------------------------------------------------------------
    # Conversion helpers (flow_pred → x0 / x_bound) — match SFP semantics.
    # ------------------------------------------------------------
    def _convert_flow_pred_to_x0(self, flow_pred, xt, timestep):
        original_dtype = flow_pred.dtype
        sigmas = self.scheduler.get_train_sigmas(flow_pred.device).double()
        timesteps = self.scheduler.get_train_timesteps(flow_pred.device).double()
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.double().unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0 = xt.double() - sigma_t * flow_pred.double()
        return x0.to(original_dtype)

    def _convert_flow_pred_to_x_bound(self, flow_pred, xt, timestep):
        original_dtype = flow_pred.dtype
        sigmas = self.scheduler.get_train_sigmas(flow_pred.device).double()
        timesteps = self.scheduler.get_train_timesteps(flow_pred.device).double()
        bound = self.timestep_bound.to(flow_pred.device).double()
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.double().unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        timestep_id_bound = torch.argmin(
            (timesteps.unsqueeze(0) - bound.unsqueeze(1)).abs(), dim=1
        )
        sigma_t_bound = sigmas[timestep_id_bound].reshape(-1, 1, 1, 1)
        x_bound = xt.double() - (sigma_t - sigma_t_bound) * flow_pred.double()
        return x_bound.to(original_dtype)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(
        self,
        noisy_image_or_video: torch.Tensor,           # [B, F, C, H, W] (SFP layout)
        conditional_dict: dict,                       # {"prompt_embeds": [B, L, D]}
        timestep_id: torch.Tensor,                    # [B, F] long, indexes the train_timesteps table
        y: Optional[torch.Tensor] = None,             # I2V first-frame embed [B, 20, F, H, W] (BCFHW)
        audio_latents: Optional[torch.Tensor] = None, # [B, A_dim, A_T]
        audio_timestep_id: Optional[torch.Tensor] = None,
        cp_mesh=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pipe = self.pipeline

        # --- timestep mapping (table -> values) ---
        train_timesteps = self.scheduler.get_train_timesteps(noisy_image_or_video.device)
        timestep_vals = train_timesteps[timestep_id]   # [B, F]
        input_timestep = timestep_vals[:, 0]           # [B] (uniform across F for bidirectional)

        # SFP layout [B, F, C, H, W] → MOVA layout [B, C, F, H, W]
        visual_bcfhw = noisy_image_or_video.permute(0, 2, 1, 3, 4).contiguous()

        # First-frame condition. The pipeline's inference_single_step expects a
        # tensor `y` with 20 channels (4 mask + 16 vae) when require_vae_embedding.
        if y is None:
            B, _C, F_lat, H_lat, W_lat = visual_bcfhw.shape
            y = torch.zeros(
                B, 20, F_lat, H_lat, W_lat,
                device=visual_bcfhw.device, dtype=visual_bcfhw.dtype,
            )

        # Audio condition. For T2V distillation we feed zero audio latents so the
        # bridge has a well-defined input but contributes nothing meaningful. The
        # audio_dit branch is frozen, so this is gradient-safe.
        if audio_latents is None:
            B = visual_bcfhw.shape[0]
            audio_dim = pipe.audio_vae.latent_dim
            audio_steps = getattr(pipe, "_distill_audio_steps", 403)
            audio_latents = torch.zeros(
                B, audio_dim, audio_steps,
                device=visual_bcfhw.device, dtype=visual_bcfhw.dtype,
            )
        if audio_timestep_id is None:
            audio_timestep_id = timestep_id
        audio_timestep_vals = train_timesteps[audio_timestep_id]
        audio_input_timestep = audio_timestep_vals[:, 0]

        context = conditional_dict["prompt_embeds"]
        video_fps = getattr(pipe, "_distill_video_fps", 24.0)

        # Pick the active DiT for this forward pass.
        active_high = self.video_dit_high
        active_low = self.video_dit_low
        active_visual_dit = active_high if self.target == "high_noise" else active_low

        # Swap the master's video DiT references so that inference_single_step
        # uses *our* DiTs. We restore on exit (also on exception).
        prev_high, prev_low = swap_video_dit(pipe, high=active_high, low=active_low)
        try:
            flow_pred_bcfhw, _audio_pred = pipe.inference_single_step(
                visual_dit=active_visual_dit,
                visual_latents=visual_bcfhw,
                audio_latents=audio_latents,
                y=y,
                context=context,
                timestep=input_timestep,
                audio_timestep=audio_input_timestep,
                video_fps=video_fps,
                cp_mesh=cp_mesh,
            )
        finally:
            swap_video_dit(pipe, high=prev_high, low=prev_low)

        # MOVA layout → SFP layout
        flow_pred = flow_pred_bcfhw.permute(0, 2, 1, 3, 4).contiguous()

        if self.target == "high_noise":
            x_pred = self._convert_flow_pred_to_x_bound(
                flow_pred=flow_pred.flatten(0, 1),
                xt=noisy_image_or_video.flatten(0, 1),
                timestep=timestep_vals.flatten(0, 1),
            ).unflatten(0, flow_pred.shape[:2])
        else:
            x_pred = self._convert_flow_pred_to_x0(
                flow_pred=flow_pred.flatten(0, 1),
                xt=noisy_image_or_video.flatten(0, 1),
                timestep=timestep_vals.flatten(0, 1),
            ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, x_pred

    # ------------------------------------------------------------
    # FSDP grad clip helper
    # ------------------------------------------------------------
    def clip_grad_norm_(self, max_norm: float):
        if hasattr(self.model, "clip_grad_norm_"):
            return self.model.clip_grad_norm_(max_norm)
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
