"""
models/losses_mri.py

MRI loss head wrapping TinyRecursiveReasoningModel_MRI.

Differences from ACTLossHead in losses.py:
  - Reconstruction loss  : MSE instead of cross-entropy
  - Accuracy metric      : PSNR instead of exact-accuracy
  - Q-halt target        : "is this prediction below per-batch-median MSE?"
  - Q-halt loss          : sum-reduced (no double normalisation by batch size)
  - Q-continue loss      : computed only when the model returns
                            ``target_q_continue`` (i.e. no_ACT_continue=False),
                            mirroring losses.ACTLossHead.

Reduction convention
--------------------
pretrain.py expects the **sum** form for the gradient term (it divides by
``global_batch_size`` once).  All loss terms emitted from this module are
therefore sums over the batch, never per-sample means.  Metrics (psnr, mse,
steps) follow the upstream TRM convention of "running sum + count" so
distributed reductions can recover proper averages by ``sum(x) / sum(count)``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MRILossHead(nn.Module):
    def __init__(self, model: nn.Module, q_halt_loss_weight: float = 0.5) -> None:
        super().__init__()
        self.model = model
        self.q_halt_loss_weight = float(q_halt_loss_weight)

    # ── Carry delegation ────────────────────────────────────────────────────

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        carry: Any,
        batch: Dict[str, torch.Tensor],
        return_keys: Sequence[str] = (),
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns
        -------
        new_carry, total_loss, metrics, detached_outputs, all_finished
        """
        new_carry, outputs = self.model(carry=carry, batch=batch)

        pred_image   = outputs["pred_image"]                                   # (B, H*W)
        target_image = new_carry.current_data["labels"].to(pred_image.dtype)   # (B, H*W)

        # ── Reconstruction loss (MSE, sum over batch) ────────────────────
        # pretrain.py divides by global_batch_size once, so this returns
        # the *sum* of per-sample MSEs.
        mse_per_sample = F.mse_loss(pred_image, target_image, reduction="none").mean(dim=-1)  # (B,)
        mse_loss       = mse_per_sample.sum()                                   # scalar

        # ── PSNR metric (halted elements only) ───────────────────────────
        halted      = new_carry.halted                              # (B,) bool
        halted_f    = halted.float()
        valid_count = halted_f.sum()

        with torch.no_grad():
            # PSNR with assumed peak signal value 1.0 (labels are normalised to [0,1]).
            psnr_per_sample = 20.0 * torch.log10(
                1.0 / mse_per_sample.detach().clamp_min(1e-8).sqrt()
            )                                                       # (B,)

            psnr_sum = (psnr_per_sample * halted_f).sum()
            mse_sum  = (mse_per_sample.detach() * halted_f).sum()
            steps_sum = (new_carry.steps.float() * halted_f).sum()

        # ── Q-halt loss ──────────────────────────────────────────────────
        # Target: 1 if this sample's MSE is below the per-batch median (good
        # prediction → safe to halt), else 0.  Only halted samples contribute.
        q_halt_logits = outputs["q_halt_logits"]                    # (B,)

        with torch.no_grad():
            median_mse = mse_per_sample.detach().median()
            q_target   = (mse_per_sample.detach() < median_mse).float()    # (B,)

        # Sum reduction: pretrain.py performs the global-batch normalisation.
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits, q_target, weight=halted_f, reduction="sum"
        )

        # ── Q-continue loss (only when ACT_continue is enabled) ──────────
        # Mirrors losses.ACTLossHead: BCE between q_continue_logits and the
        # bootstrap target_q_continue produced by the model under no_grad.
        q_continue_loss = pred_image.new_zeros(())
        if "target_q_continue" in outputs:
            q_continue_logits = outputs["q_continue_logits"]        # (B,)
            target_q_continue = outputs["target_q_continue"].detach().to(q_continue_logits.dtype)
            q_continue_loss = F.binary_cross_entropy_with_logits(
                q_continue_logits, target_q_continue, weight=halted_f, reduction="sum"
            )

        # ── Total loss ───────────────────────────────────────────────────
        total_loss = mse_loss + self.q_halt_loss_weight * (q_halt_loss + q_continue_loss)

        # ── Metrics dict (all tensors, reduced in pretrain.py) ───────────
        metrics: Dict[str, torch.Tensor] = {
            "count":           valid_count,                          # (scalar)
            "mse":             mse_sum,
            "psnr":            psnr_sum,
            "steps":           steps_sum,
            "mse_loss":        mse_loss.detach(),
            "q_halt_loss":     q_halt_loss.detach(),
            "q_continue_loss": q_continue_loss.detach(),
        }

        # ── Detach selected outputs for evaluators / logging ─────────────
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        all_finished: torch.Tensor = new_carry.halted.all()

        return new_carry, total_loss, metrics, detached_outputs, all_finished
