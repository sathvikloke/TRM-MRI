"""
models/ema.py

Exponential-moving-average helper for TRM training.

Original upstream version only tracked named_parameters() with
requires_grad=True.  In TRM, several pieces of stateful tensors live in
persistent nn.Buffer objects rather than nn.Parameter, e.g.:

    * H_init / L_init      (recursive_reasoning/trm_mri.py)
    * sparse-embedding weights (sparse_embedding.CastedSparseEmbedding)

If those buffers ever change during training (sparse-emb weights do, via
the SignSGD optimizer hook), the upstream EMA silently drops them and
the EMA-evaluated checkpoint diverges from the live model. We therefore
also track *persistent floating-point buffers* and update them with the
same EMA recurrence used for parameters.

Non-floating-point buffers (int32 ids, bool masks) are skipped — averaging
them is meaningless. Non-persistent buffers (RoPE cos/sin caches,
local_weights, local_ids) are also skipped because they are recomputed
every forward pass.
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn


def _unwrap(module: nn.Module) -> nn.Module:
    if isinstance(module, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return module.module
    return module


def _persistent_buffer_names(module: nn.Module) -> set[str]:
    """Return the set of fully-qualified buffer names that are persistent."""
    persistent: set[str] = set()
    for submodule_name, submodule in module.named_modules():
        # _non_persistent_buffers_set is part of nn.Module's public-ish API
        non_persistent = getattr(submodule, "_non_persistent_buffers_set", set())
        for buf_name, _ in submodule.named_buffers(recurse=False):
            if buf_name in non_persistent:
                continue
            full = f"{submodule_name}.{buf_name}" if submodule_name else buf_name
            persistent.add(full)
    return persistent


class EMAHelper:
    """
    Tracks an EMA of (a) all trainable parameters and (b) all persistent
    floating-point buffers in `module`.  Compatible with the previous
    EMAHelper API: register/update/ema/ema_copy/state_dict/load_state_dict.
    """

    def __init__(self, mu: float = 0.999) -> None:
        if not 0.0 <= mu <= 1.0:
            raise ValueError(f"EMA decay mu must be in [0, 1]; got {mu}.")
        self.mu = mu
        self.shadow: Dict[str, torch.Tensor] = {}

    # ── name iterators ──────────────────────────────────────────────────────
    def _iter_tracked(self, module: nn.Module):
        """Yield (name, tensor) for every parameter/buffer we EMA-track."""
        module = _unwrap(module)
        # Parameters – include even requires_grad=False to be safe; if a user
        # later un-freezes a layer mid-training, the shadow already exists.
        for name, param in module.named_parameters():
            yield name, param.data
        # Persistent floating-point buffers
        persistent = _persistent_buffer_names(module)
        for name, buf in module.named_buffers():
            if name not in persistent:
                continue
            if not buf.is_floating_point():
                continue
            yield name, buf.data

    # ── public API ──────────────────────────────────────────────────────────
    def register(self, module: nn.Module) -> None:
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in self._iter_tracked(module)
        }

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, tensor in self._iter_tracked(module):
            if name not in self.shadow:
                # New tensor appeared after register() (e.g. lazy init).
                self.shadow[name] = tensor.detach().clone()
                continue
            shadow = self.shadow[name]
            if shadow.shape != tensor.shape or shadow.device != tensor.device:
                # Shape/device drift – re-anchor the shadow.
                self.shadow[name] = tensor.detach().clone()
                continue
            # shadow ← mu * shadow + (1 - mu) * live
            shadow.mul_(self.mu).add_(tensor, alpha=1.0 - self.mu)

    @torch.no_grad()
    def ema(self, module: nn.Module) -> None:
        """Copy the EMA shadow into `module` in-place."""
        module = _unwrap(module)
        # Parameters
        for name, param in module.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        # Persistent floating-point buffers
        persistent = _persistent_buffer_names(module)
        for name, buf in module.named_buffers():
            if name not in persistent or not buf.is_floating_point():
                continue
            if name in self.shadow:
                buf.data.copy_(self.shadow[name])

    def ema_copy(self, module: nn.Module) -> nn.Module:
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    # ── checkpointing ───────────────────────────────────────────────────────
    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        # Clone tensors so external mutation can't poison the shadow.
        self.shadow = {k: v.detach().clone() for k, v in state_dict.items()}
