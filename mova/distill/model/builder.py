"""
Factory for building the multi-DiT MOVA distillation graph.

The DMD distillation needs three (or four) *independent* video DiT copies that
all share the heavyweight peripherals (text_encoder, audio_dit, dual_tower_bridge,
video_vae, audio_vae, scheduler):

    generator         : trainable student     (video_dit + video_dit_2 owned)
    real_score        : frozen teacher        (video_dit + video_dit_2 owned)
    fake_score        : trainable critic      (video_dit + video_dit_2 owned)
    high_noise_teacher: frozen distilled high-noise model used only in low-noise stage

We achieve this by:
    1. Loading one MOVATrain pipeline `master` from the pretrained checkpoint.
    2. Cloning ONLY the video_dit / video_dit_2 weights into a separate
       `MOVATrain`-shaped facade for each role.
    3. Sharing the master's `text_encoder`, `audio_dit`, `dual_tower_bridge`,
       `video_vae`, `audio_vae`, and `scheduler` by reference.

Because video_dit is the only heavyweight component duplicated across roles, the
peak memory cost of the distillation graph is roughly (1 master + 3 extra video_dits).
With FSDP full-sharding across 8 H100s this is feasible at 720p/81f.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from mova.diffusion.pipelines.mova_train import MOVATrain, MOVATrain_from_pretrained


@dataclass
class MOVADistillModules:
    """Container holding the four DiT roles plus the shared MOVATrain master."""
    master: MOVATrain
    generator_high: nn.Module       # trainable, alias of master.video_dit
    generator_low: nn.Module        # trainable, alias of master.video_dit_2
    real_high: nn.Module            # frozen teacher (deep-copied)
    real_low: nn.Module             # frozen teacher (deep-copied)
    fake_high: nn.Module            # trainable critic (deep-copied)
    fake_low: nn.Module             # trainable critic (deep-copied)
    # high_noise_teacher is filled in lazily for the second (low) stage; it is
    # the *generator_high* state captured at the end of the first stage.
    distilled_high_state: Optional[dict] = None


def _deepcopy_dit(module: nn.Module) -> nn.Module:
    """Deep-copy a video DiT, then unconditionally freeze it."""
    cloned = copy.deepcopy(module)
    cloned.requires_grad_(False)
    cloned.eval()
    return cloned


def _trainable_clone_dit(module: nn.Module) -> nn.Module:
    """Deep-copy a video DiT for training (parameters new, but we keep .train()
    state managed by the trainer)."""
    cloned = copy.deepcopy(module)
    cloned.requires_grad_(True)
    return cloned


def build_distill_modules(
    pretrained_path: str,
    use_gradient_checkpointing: bool = True,
    use_gradient_checkpointing_offload: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> MOVADistillModules:
    """Load one MOVATrain pipeline from `pretrained_path` and produce the DMD
    distillation module set. The returned `master` pipeline acts as the
    *generator*; `real_*` and `fake_*` are deep-copied to be independent.

    Note: text_encoder / audio_dit / dual_tower_bridge / video_vae / audio_vae
    remain referenced exclusively by `master`. The wrapper code in
    `mova_dit_wrapper.py` swaps in the correct video_dit at forward time.
    """
    master: MOVATrain = MOVATrain_from_pretrained(
        from_pretrained=pretrained_path,
        device=device,
        torch_dtype=torch_dtype,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )

    # --- build clones ---
    real_high = _deepcopy_dit(master.video_dit)
    real_low = _deepcopy_dit(master.video_dit_2)

    fake_high = _trainable_clone_dit(master.video_dit)
    fake_low = _trainable_clone_dit(master.video_dit_2)

    # --- generator points at master.video_dit / video_dit_2 (the trainable ones) ---
    master.video_dit.requires_grad_(True)
    master.video_dit_2.requires_grad_(True)

    # The dual_tower_bridge and audio_dit are FROZEN during distillation (we are
    # not retraining the cross-modal interaction; we only want the student video
    # branches to match the teacher distribution).
    master.dual_tower_bridge.requires_grad_(False)
    master.audio_dit.requires_grad_(False)
    master.text_encoder.requires_grad_(False)
    master.video_vae.requires_grad_(False)
    master.audio_vae.requires_grad_(False)

    return MOVADistillModules(
        master=master,
        generator_high=master.video_dit,
        generator_low=master.video_dit_2,
        real_high=real_high,
        real_low=real_low,
        fake_high=fake_high,
        fake_low=fake_low,
    )


def swap_video_dit(pipeline: MOVATrain, *, high: nn.Module, low: nn.Module):
    """Temporarily swap the video DiT modules of a master pipeline. Returns the
    previous (high, low) so the caller can restore them.

    This is used by the wrapper to route forward through a specific role's DiT
    while keeping the rest of MOVATrain.inference_single_step's flow intact."""
    prev_high = pipeline.video_dit
    prev_low = pipeline.video_dit_2
    pipeline.video_dit = high
    pipeline.video_dit_2 = low
    return prev_high, prev_low
