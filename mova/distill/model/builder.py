"""
Factory for building the MOVA distillation graph with single-model loading.

When training in "high_noise" mode, only ``video_dit`` (high) is loaded and cloned.
When training in "low_noise" mode,  only ``video_dit_2`` (low) is loaded and cloned.

There is no centralized master pipeline. Each MOVAVideoDiTWrapper directly owns
its single DiT and references shared resources (audio_dit, dual_tower_bridge,
scheduler, etc.) for its self-contained inference.

    generator  : trainable student     — alias of the pretrained DiT
    real_score : frozen teacher        — deep-copied from pretrained DiT
    fake_score : trainable critic      — deep-copied from pretrained DiT
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn

from mova.diffusion.pipelines.mova_train import MOVATrain, MOVATrain_from_pretrained, prompt_clean
from mova.distill.utils.scheduler_distill import FlowMatchSchedulerDistill


@dataclass
class MOVADistillModules:
    """Container with single-model DiT roles + independent shared resources.

    No centralized master pipeline — each role accesses shared resources
    (audio_dit, dual_tower_bridge, etc.) by direct reference, and each
    MOVAVideoDiTWrapper owns its inference logic internally.
    """
    training_target: str
    generator_dit: nn.Module
    real_dit: nn.Module
    fake_dit: nn.Module

    audio_dit: nn.Module
    dual_tower_bridge: nn.Module
    text_encoder: nn.Module
    text_encoder_2: Optional[nn.Module]
    tokenizer: object
    video_vae: nn.Module
    audio_vae: nn.Module
    scheduler: FlowMatchSchedulerDistill

    use_gradient_checkpointing: bool
    use_gradient_checkpointing_offload: bool


def _deepcopy_dit(module: nn.Module) -> nn.Module:
    cloned = copy.deepcopy(module)
    cloned.requires_grad_(False)
    cloned.eval()
    return cloned


def _trainable_clone_dit(module: nn.Module) -> nn.Module:
    cloned = copy.deepcopy(module)
    cloned.requires_grad_(True)
    return cloned


def build_distill_modules(
    pretrained_path: str,
    training_target: str = "high_noise",
    use_gradient_checkpointing: bool = True,
    use_gradient_checkpointing_offload: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> MOVADistillModules:
    """Load ONE MOVATrain pipeline, extract shared resources, clone the active
    DiT (*only* the one matching `training_target`), then discard the master.

    Args:
        training_target: "high_noise" → loads ``master.video_dit``;
                         "low_noise"  → loads ``master.video_dit_2``.
    """
    assert training_target in {"high_noise", "low_noise"}

    master: MOVATrain = MOVATrain_from_pretrained(
        from_pretrained=pretrained_path,
        device=device,
        torch_dtype=torch_dtype,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )

    orig = master.scheduler
    scheduler = FlowMatchSchedulerDistill(
        num_train_timesteps=getattr(orig, "num_train_timesteps", 1000),
        shift=getattr(orig, "shift", 5.0),
    )

    audio_dit = master.audio_dit
    dual_tower_bridge = master.dual_tower_bridge
    text_encoder = master.text_encoder
    text_encoder_2 = getattr(master, "text_encoder_2", None)
    tokenizer = getattr(master, "tokenizer", None)
    video_vae = master.video_vae
    audio_vae = master.audio_vae

    if training_target == "high_noise":
        pretrained_dit = master.video_dit
    else:
        pretrained_dit = master.video_dit_2

    real_dit = _deepcopy_dit(pretrained_dit)
    fake_dit = _trainable_clone_dit(pretrained_dit)

    generator_dit = pretrained_dit
    generator_dit.requires_grad_(True)

    dual_tower_bridge.requires_grad_(False)
    audio_dit.requires_grad_(False)
    text_encoder.requires_grad_(False)
    video_vae.requires_grad_(False)
    audio_vae.requires_grad_(False)

    return MOVADistillModules(
        training_target=training_target,
        generator_dit=generator_dit,
        real_dit=real_dit,
        fake_dit=fake_dit,
        audio_dit=audio_dit,
        dual_tower_bridge=dual_tower_bridge,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        video_vae=video_vae,
        audio_vae=audio_vae,
        scheduler=scheduler,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )


def encode_text_prompts(
    prompts: Union[str, List[str]],
    tokenizer,
    text_encoder: nn.Module,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    max_sequence_length: int = 512,
    num_videos_per_prompt: int = 1,
) -> torch.Tensor:
    dtype = dtype or text_encoder.dtype

    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    prompts = [prompt_clean(u) for u in prompts]

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    mask = text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(len(prompts) * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def normalize_video_latents(video_vae: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(video_vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
        1, video_vae.config.z_dim, 1, 1, 1
    )
    inv_std = (1.0 / torch.tensor(video_vae.config.latents_std, device=latents.device, dtype=latents.dtype)).view(
        1, video_vae.config.z_dim, 1, 1, 1
    )
    return (latents - mean) * inv_std