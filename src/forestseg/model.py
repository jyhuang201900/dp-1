from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        if mask is not None:
            prob = prob * mask
            target = target * mask
        inter = torch.sum(prob * target)
        denom = torch.sum(prob) + torch.sum(target) + self.eps
        return 1.0 - (2.0 * inter + self.eps) / denom


def build_model(encoder_name: str = "resnet34", in_channels: int = 1, classes: int = 1) -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )


def to_device(model: nn.Module) -> tuple[nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device


def enable_mc_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()
