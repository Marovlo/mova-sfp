"""
FSDP utilities for the MOVA DMD distillation trainer. Direct port of
Self-Forcing-Plus's `utils/distributed.py`, which uses raw PyTorch FSDP1.

Why not use Accelerate's FullyShardedDataParallelPlugin like the rest of MOVA?
The DMD trainer needs to wrap *multiple* large models independently
(generator / real_score / fake_score / high_noise_teacher / text_encoder) and
have fine-grained per-module control of:
  * sharding strategy
  * mixed precision policy
  * cpu_offload (per-module)
  * grad clip helpers (clip_grad_norm_ on the FSDP root)
Accelerate.prepare() assumes a single model. Doing it manually is the simpler,
better-documented path here.
"""

from __future__ import annotations

import os
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])
    init_method = (
        f"tcp://[{host}]:{port}" if ":" in host else f"tcp://{host}:{port}"
    )
    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size,
        init_method=init_method, timeout=timedelta(minutes=60),
    )
    torch.cuda.set_device(local_rank)


def fsdp_wrap(
    module,
    sharding_strategy: str = "full",
    mixed_precision: bool = True,
    wrap_strategy: str = "size",
    min_num_params: int = int(5e7),
    transformer_module=None,
    cpu_offload: bool = False,
    ignored_modules=None,
):
    if mixed_precision:
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mp = None

    if wrap_strategy == "transformer":
        assert transformer_module is not None, "transformer wrap needs `transformer_module`"
        policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_module)
    elif wrap_strategy == "size":
        policy = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    else:
        raise ValueError(f"Unknown wrap_strategy: {wrap_strategy}")

    os.environ.setdefault("NCCL_CROSS_NIC", "1")
    strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    return FSDP(
        module,
        auto_wrap_policy=policy,
        sharding_strategy=strategy,
        mixed_precision=mp,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        ignored_modules=ignored_modules,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False,
    )


def fsdp_state_dict(model):
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        return model.state_dict()


class EMA_FSDP:
    """Float-on-CPU exponential moving average of an FSDP-wrapped module's
    full parameter set. Synchronisation is performed at update time via
    `summon_full_params`."""

    def __init__(self, fsdp_module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict = {}
        with FSDP.summon_full_params(fsdp_module, writeback=False, offload_to_cpu=True, rank0_only=True):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        with FSDP.summon_full_params(fsdp_module, writeback=False, offload_to_cpu=True, rank0_only=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    self.shadow[n].mul_(self.decay).add_(p.detach().float().cpu(), alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}
