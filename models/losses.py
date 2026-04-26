"""
models/losses.py

ACT loss head for the Samsung TRM ARC training pipeline.
Mirrors the upstream TRM ACTLossHead so cfg_pretrain.yaml resolves correctly.
For the MRI pipeline see models/losses_mri.py instead.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
) -> torch.Tensor:
    """
    Numerically stable cross-entropy with ignore_index masking.
    Returns per-token loss with shape == labels.shape.
    """
    logits = logits.to(torch.float32)
    valid_mask = (labels != ignore_index)
    safe_labels = labels.clone()
    safe_labels[~valid_mask] = 0
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1).long()).squeeze(-1)
    return nll * valid_mask.float()


def softmax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_LABEL_ID,
) -> torch.Tensor:
    """Standard cross-entropy with ignore_index masking. Per-token loss."""
    return F.cross_entropy(
        logits.flatten(0, -2).to(torch.float32),
        labels.flatten().long(),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape)


LOSS_FNS = {
    "stablemax_cross_entropy": stablemax_cross_entropy,
    "softmax_cross_entropy":   softmax_cross_entropy,
}


class ACTLossHead(nn.Module):
    """
    Wraps a TinyRecursiveReasoningModel_ACTV1 and produces:
      - LM cross-entropy loss
      - Q-halt BCE loss   (target = "is this prediction correct?")
      - Q-continue BCE loss when no_ACT_continue=False
    Returns (new_carry, total_loss, metrics, detached_outputs, all_finished).
    """

    def __init__(self, model: nn.Module, loss_type: str = "softmax_cross_entropy") -> None:
        super().__init__()
        self.model = model
        if loss_type not in LOSS_FNS:
            raise ValueError(f"Unknown loss_type: {loss_type!r}. Choose from {list(LOSS_FNS)}.")
        self.loss_fn = LOSS_FNS[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        carry: Any,
        batch: Dict[str, torch.Tensor],
        return_keys: Sequence[str] = (),
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:

        new_carry, outputs = self.model(carry=carry, batch=batch)

        labels = new_carry.current_data["labels"]
        logits = outputs["logits"]

        # ── LM loss (sum reduction; pretrain.py scales by 1/global_batch_size) ─
        loss_per_token = self.loss_fn(logits, labels)
        lm_loss = loss_per_token.sum()

        # ── Token / sequence accuracy ─
        with torch.no_grad():
            valid_mask = (labels != IGNORE_LABEL_ID)
            preds = logits.argmax(dim=-1)
            correct_per_token = (preds == labels) & valid_mask
            valid_per_seq     = valid_mask.sum(dim=-1).clamp_min(1)
            seq_correct       = (correct_per_token.sum(dim=-1) == valid_per_seq)

            halted      = new_carry.halted
            valid_count = halted.sum().clamp_min(1)

            accuracy_sum = torch.where(halted, seq_correct.float(), torch.zeros_like(seq_correct, dtype=torch.float32)).sum()
            steps_sum    = torch.where(halted, new_carry.steps.float(), torch.zeros_like(new_carry.steps, dtype=torch.float32)).sum()

        # ── Q-halt loss ─
        q_halt_logits = outputs["q_halt_logits"]
        with torch.no_grad():
            q_halt_target = seq_correct.float()
            weight        = halted.float()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits, q_halt_target, weight=weight, reduction="sum"
        )

        # ── Q-continue loss (only when ACT continue is enabled) ─
        q_continue_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
        if "target_q_continue" in outputs:
            q_continue_logits = outputs["q_continue_logits"]
            q_continue_loss = F.binary_cross_entropy_with_logits(
                q_continue_logits, outputs["target_q_continue"], weight=weight, reduction="sum"
            )

        total_loss = lm_loss + 0.5 * q_halt_loss + 0.5 * q_continue_loss

        metrics: Dict[str, torch.Tensor] = {
            "count":       valid_count.float(),
            "accuracy":    accuracy_sum,
            "steps":       steps_sum,
            "lm_loss":     lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        }
        if "target_q_continue" in outputs:
            metrics["q_continue_loss"] = q_continue_loss.detach()

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        all_finished: torch.Tensor = new_carry.halted.all()

        return new_carry, total_loss, metrics, detached_outputs, all_finished
