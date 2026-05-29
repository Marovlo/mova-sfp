"""
MOVA single-model Video-DiT wrapper for SFP-style DMD distillation.

A wrapper presents one role (generator / real_score / fake_score) as a self-
contained module with the SFP `WanDiffusionWrapper` interface
`forward(noisy_image_or_video, conditional_dict, timestep_id, ...) -> (flow_pred, x_pred)`.

Key differences from the old dual-model architecture:
  * Wraps a SINGLE DiT (no high/low pair).  The training_target is selected at
    launch time and only the corresponding DiT is loaded.
  * No master pipeline — shared resources (audio_dit, dual_tower_bridge, scheduler)
    are passed by reference. Inference logic lives inside the wrapper itself.
  * No swap_video_dit — each wrapper directly accesses its own DiT.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from mova.diffusion.models import sinusoidal_embedding_1d
from mova.distributed.functional import (
    _sp_all_gather_avg,
    _sp_split_tensor,
    _sp_split_tensor_dim_0,
)
from mova.distill.utils.scheduler_distill import FlowMatchSchedulerDistill


class MOVAVideoDiTWrapper(nn.Module):
    """SFP-style wrapper around a SINGLE video DiT with self-contained inference.

    Args:
        video_dit : the role's DiT.
        target    : "high_noise" or "low_noise" (determines x0 vs x_bound conversion).
        boundary_step  : raw boundary step (0..1000) used to compute x_bound.
        timestep_shift : sigma shift used by the FlowMatchScheduler.
        audio_dit     : shared frozen audio DiT.
        dual_tower_bridge : shared frozen dual-tower bridge.
        scheduler     : FlowMatchSchedulerDistill instance.
        use_gradient_checkpointing        : forward via checkpoint.
        use_gradient_checkpointing_offload: offload activations to CPU.
    """

    def __init__(
        self,
        video_dit: nn.Module,
        target: str = "high_noise",
        boundary_step: int = 900,
        timestep_shift: float = 5.0,
        audio_dit: Optional[nn.Module] = None,
        dual_tower_bridge: Optional[nn.Module] = None,
        scheduler: Optional[FlowMatchSchedulerDistill] = None,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = True,
    ):
        super().__init__()
        assert target in {"high_noise", "low_noise"}

        self.target = target
        self.timestep_shift = timestep_shift
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload

        self.add_module("video_dit", video_dit)
        self.model = self.video_dit

        self.audio_dit = audio_dit
        self.dual_tower_bridge = dual_tower_bridge
        self.scheduler = scheduler

        self.timestep_bound = torch.tensor([boundary_step], dtype=torch.float64)
        if timestep_shift > 1:
            self.timestep_bound = (
                timestep_shift
                * (self.timestep_bound / 1000.0)
                / (1 + (timestep_shift - 1) * (self.timestep_bound / 1000.0))
                * 1000.0
            )

    def enable_gradient_checkpointing(self) -> None:
        return

    # ------------------------------------------------------------
    # Conversion helpers
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
    # Self-contained inference (migrated from MOVATrain)
    # ------------------------------------------------------------
    def _forward_dual_tower_dit(
        self,
        visual_dit,
        visual_x: torch.Tensor,
        audio_x: torch.Tensor,
        visual_context: torch.Tensor,
        audio_context: torch.Tensor,
        visual_t_mod: torch.Tensor,
        audio_t_mod: Optional[torch.Tensor],
        visual_freqs: torch.Tensor,
        audio_freqs: torch.Tensor,
        grid_size: tuple,
        video_fps: float,
        condition_scale: Optional[float] = 1.0,
        a2v_condition_scale: Optional[float] = None,
        v2a_condition_scale: Optional[float] = None,
        cp_mesh: Optional[DeviceMesh] = None,
    ):
        min_layers = min(len(visual_dit.blocks), len(self.audio_dit.blocks))
        visual_layers = len(visual_dit.blocks)

        sp_enabled = False
        sp_group = None
        sp_rank = 0
        sp_size = 1
        visual_pad_len = 0
        audio_pad_len = 0

        if self.dual_tower_bridge.apply_cross_rope:
            (visual_rope_cos_sin, audio_rope_cos_sin) = self.dual_tower_bridge.build_aligned_freqs(
                video_fps=video_fps,
                grid_size=grid_size,
                audio_steps=audio_x.shape[1],
                device=visual_x.device,
                dtype=visual_x.dtype,
            )
        else:
            visual_rope_cos_sin = None
            audio_rope_cos_sin = None

        if cp_mesh is not None:
            sp_rank = cp_mesh.get_local_rank()
            sp_size = cp_mesh.size()
            sp_group = cp_mesh.get_group()
            visual_x, visual_chunk_len, visual_pad_len, _ = _sp_split_tensor(visual_x, sp_size=sp_size, sp_rank=sp_rank)
            audio_x, audio_chunk_len, audio_pad_len, _ = _sp_split_tensor(audio_x, sp_size=sp_size, sp_rank=sp_rank)
            visual_freqs, _, _, _ = _sp_split_tensor_dim_0(visual_freqs, sp_size=sp_size, sp_rank=sp_rank)
            audio_freqs, _, _, _ = _sp_split_tensor_dim_0(audio_freqs, sp_size=sp_size, sp_rank=sp_rank)
            if visual_rope_cos_sin is not None:
                visual_rope_cos_sin = [
                    _sp_split_tensor(rope_cos_sin, sp_size=sp_size, sp_rank=sp_rank)[0]
                    for rope_cos_sin in visual_rope_cos_sin
                ]
            if audio_rope_cos_sin is not None:
                audio_rope_cos_sin = [
                    _sp_split_tensor(rope_cos_sin, sp_size=sp_size, sp_rank=sp_rank)[0]
                    for rope_cos_sin in audio_rope_cos_sin
                ]
            if len(visual_t_mod.shape) == 4:
                visual_t_mod, _, _, _ = _sp_split_tensor(visual_t_mod, sp_size=sp_size, sp_rank=sp_rank)
            sp_enabled = True

        def _make_custom_forward(module):
            def _fn(*inputs):
                return module(*inputs)
            return _fn

        self.audio_dit.eval()
        self.dual_tower_bridge.eval()

        for layer_idx in range(min_layers):
            visual_block = visual_dit.blocks[layer_idx]
            audio_block = self.audio_dit.blocks[layer_idx]

            # ========== Bridge 交互 (无需梯度) ==========
            if self.dual_tower_bridge.should_interact(layer_idx, 'a2v'):
                with torch.no_grad():
                    visual_x_detached = visual_x.detach()
                    audio_x_detached = audio_x.detach()
                    
                    bridge_out_visual, bridge_out_audio = self.dual_tower_bridge(
                        layer_idx,
                        visual_x_detached,
                        audio_x_detached,
                        x_freqs=visual_rope_cos_sin,
                        y_freqs=audio_rope_cos_sin,
                        a2v_condition_scale=a2v_condition_scale,
                        v2a_condition_scale=v2a_condition_scale,
                        condition_scale=condition_scale,
                        video_grid_size=grid_size,
                    )
                
                visual_x = visual_x + (bridge_out_visual - visual_x_detached)
                audio_x = bridge_out_audio

            # ========== Visual Block (需要梯度 + checkpoint) ==========
            if self.use_gradient_checkpointing and torch.is_grad_enabled():
                if self.use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        print(f"[{layer_idx}/{min_layers}] layer visual_block")
                        visual_x = torch.utils.checkpoint.checkpoint(
                            _make_custom_forward(visual_block),
                            visual_x, visual_context, visual_t_mod, visual_freqs,
                            use_reentrant=False,
                        )
                else:
                    visual_x = torch.utils.checkpoint.checkpoint(
                        _make_custom_forward(visual_block),
                        visual_x, visual_context, visual_t_mod, visual_freqs,
                        use_reentrant=False,
                    )
            else:
                visual_x = visual_block(visual_x, visual_context, visual_t_mod, visual_freqs)

            # ========== Audio Block (无需梯度，直接前向) ==========
            with torch.no_grad():  # ⚡ 不保存任何激活值，不走 checkpoint/offload
                audio_x = audio_block(audio_x, audio_context, audio_t_mod, audio_freqs)


        # ========== 剩余 Visual-only 层 (保持不变) ==========
        for layer_idx in range(min_layers, visual_layers):
            visual_block = visual_dit.blocks[layer_idx]
            if self.use_gradient_checkpointing and torch.is_grad_enabled():
                if self.use_gradient_checkpointing_offload:
                    print(f"[{layer_idx}] remain layer visual_block")
                    with torch.autograd.graph.save_on_cpu():
                        visual_x = torch.utils.checkpoint.checkpoint(
                            _make_custom_forward(visual_block),
                            visual_x, visual_context, visual_t_mod, visual_freqs,
                            use_reentrant=False,
                        )
                else:
                    visual_x = torch.utils.checkpoint.checkpoint(
                        _make_custom_forward(visual_block),
                        visual_x, visual_context, visual_t_mod, visual_freqs,
                        use_reentrant=False,
                    )
            else:
                visual_x = visual_block(visual_x, visual_context, visual_t_mod, visual_freqs)

        if sp_enabled:
            visual_x_full = _sp_all_gather_avg(visual_x, sp_group=sp_group, pad_len=visual_pad_len)
            audio_x_full = _sp_all_gather_avg(audio_x, sp_group=sp_group, pad_len=audio_pad_len)
        else:
            visual_x_full = visual_x
            audio_x_full = audio_x

        return visual_x_full, audio_x_full

    def _inference_single_step(
        self,
        visual_dit,
        visual_latents: torch.Tensor,
        audio_latents: Optional[torch.Tensor],
        y,
        context: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: Optional[torch.Tensor],
        video_fps: float,
        cp_mesh=None
    ):
        audio_context = visual_context = context

        if audio_timestep is None:
            audio_timestep = timestep

        model_dtype = torch.bfloat16
        with torch.autocast("cuda", dtype=torch.float32):
            visual_t = visual_dit.time_embedding(sinusoidal_embedding_1d(visual_dit.freq_dim, timestep))
            visual_t_mod = visual_dit.time_projection(visual_t).unflatten(1, (6, visual_dit.dim))

            audio_t = self.audio_dit.time_embedding(sinusoidal_embedding_1d(self.audio_dit.freq_dim, audio_timestep))
            audio_t_mod = self.audio_dit.time_projection(audio_t).unflatten(1, (6, self.audio_dit.dim))

        visual_t = visual_t.to(model_dtype)
        visual_t_mod = visual_t_mod.to(model_dtype)
        audio_t = audio_t.to(model_dtype)
        audio_t_mod = audio_t_mod.to(model_dtype)

        visual_context_emb = visual_dit.text_embedding(visual_context)
        audio_context_emb = self.audio_dit.text_embedding(audio_context)

        visual_x = visual_latents.to(dtype=model_dtype, device=visual_latents.device)
        if audio_latents is None:
            B = visual_latents.shape[0]
            audio_in_dim = getattr(self.audio_dit.config, 'in_dim', 128)  # ← 从 audio_dit config 获取
            audio_steps = 403
            audio_latents = torch.zeros(
                B, audio_in_dim, audio_steps,
                device=visual_latents.device, dtype=model_dtype
            )
        audio_x = audio_latents.to(dtype=model_dtype, device=audio_latents.device)
        if visual_dit.require_vae_embedding:
            visual_x = torch.cat([visual_x, y], dim=1).to(dtype=model_dtype, device=visual_latents.device)

        visual_x, (t, h, w) = visual_dit.patchify(visual_x)
        grid_size = (t, h, w)

        visual_freqs = tuple(freq.to(visual_x.device) for freq in visual_dit.freqs)
        visual_freqs = torch.cat([
            visual_freqs[0][:t].view(t, 1, 1, -1).expand(t, h, w, -1),
            visual_freqs[1][:h].view(1, h, 1, -1).expand(t, h, w, -1),
            visual_freqs[2][:w].view(1, 1, w, -1).expand(t, h, w, -1)
        ], dim=-1).reshape(t * h * w, 1, -1).to(visual_x.device)

        audio_x, (f,) = self.audio_dit.patchify(audio_x, None)

        audio_freqs = torch.cat(
            [
                self.audio_dit.freqs[0][:f].view(f, -1).expand(f, -1),
                self.audio_dit.freqs[1][:f].view(f, -1).expand(f, -1),
                self.audio_dit.freqs[2][:f].view(f, -1).expand(f, -1),
            ],
            dim=-1
        ).reshape(f, 1, -1).to(audio_x.device)

        from mova.distill.model.dmd import log_cuda_memory_simple
        log_cuda_memory_simple('before _forward_dual_tower_dit')
        
        visual_x, audio_x = self._forward_dual_tower_dit(
            visual_dit=visual_dit,
            visual_x=visual_x,
            audio_x=audio_x,
            visual_context=visual_context_emb,
            audio_context=audio_context_emb,
            visual_t_mod=visual_t_mod,
            audio_t_mod=audio_t_mod,
            visual_freqs=visual_freqs,
            audio_freqs=audio_freqs,
            grid_size=grid_size,
            video_fps=video_fps,
            cp_mesh=cp_mesh,
        )
        log_cuda_memory_simple('after _forward_dual_tower_dit')

        visual_output = visual_dit.head(visual_x, visual_t)
        visual_output = visual_dit.unpatchify(visual_output, grid_size)

        audio_output = self.audio_dit.head(audio_x, audio_t)
        audio_output = self.audio_dit.unpatchify(audio_output, (f,))

        return visual_output, audio_output

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep_id: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        audio_latents: Optional[torch.Tensor] = None,
        audio_timestep_id: Optional[torch.Tensor] = None,
        cp_mesh=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        train_timesteps = self.scheduler.get_train_timesteps(noisy_image_or_video.device)
        timestep_vals = train_timesteps[timestep_id]
        input_timestep = timestep_vals[:, 0]

        visual_bcfhw = noisy_image_or_video.permute(0, 2, 1, 3, 4).contiguous()

        initial_latent = conditional_dict.get("initial_latent", None)
        if initial_latent is not None:
            first_frame_bcfhw = initial_latent.permute(0, 2, 1, 3, 4).contiguous().to(
                device=visual_bcfhw.device, dtype=visual_bcfhw.dtype)
            visual_bcfhw = torch.cat([first_frame_bcfhw, visual_bcfhw], dim=2)

        if y is None:
            B, _C, F_lat, H_lat, W_lat = visual_bcfhw.shape
            y = torch.zeros(
                B, 20, F_lat, H_lat, W_lat,
                device=visual_bcfhw.device, dtype=visual_bcfhw.dtype,
            )

        context = conditional_dict["prompt_embeds"]
        video_fps = 24.0

        if audio_timestep_id is None:
            audio_timestep_id = timestep_id
        audio_timestep_vals = train_timesteps[audio_timestep_id]
        audio_input_timestep = audio_timestep_vals[:, 0]

        flow_pred_bcfhw, _audio_pred = self._inference_single_step(
            visual_dit=self.video_dit,
            visual_latents=visual_bcfhw,
            audio_latents=audio_latents,
            y=y,
            context=context,
            timestep=input_timestep,
            audio_timestep=audio_input_timestep,
            video_fps=video_fps,
            cp_mesh=cp_mesh
        )

        flow_pred = flow_pred_bcfhw.permute(0, 2, 1, 3, 4).contiguous()

        if initial_latent is not None:
            flow_pred = flow_pred[:, 1:, ...]

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

    def clip_grad_norm_(self, max_norm: float):
        if hasattr(self.model, "clip_grad_norm_"):
            return self.model.clip_grad_norm_(max_norm)
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)