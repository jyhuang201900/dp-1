from __future__ import annotations

import torch


def spec_consistency_loss(pred_prob: torch.Tensor, spec_prob: torch.Tensor, spec_conf: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred_prob - spec_prob) * spec_conf)


def tex_consistency_loss(pred_prob: torch.Tensor, tex_prob: torch.Tensor, tex_conf: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred_prob - tex_prob) * tex_conf)


def warmup_scale(epoch_idx: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch_idx + 1) / float(warmup_epochs))
