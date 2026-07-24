"""
losses.py
=========

Multi-task loss for EdgeGuard AI: combines

    - Huber (Smooth L1) loss for the Brake Health regression head
      (more robust to outliers than plain MSE, which was implicated in
      the "regression exploding" failure mode from earlier versions)
    - Cross-entropy loss for the Brake Fade Risk classification head
    - Cross-entropy loss for the Maintenance Recommendation classification head

Losses are combined via a weighted sum, with weights sourced from
config.TrainConfig (never hardcoded here). Because the regression target
is MinMax-scaled to [0, 1] (see dataset.py), its Huber loss lives on a
comparable numeric scale to the classification cross-entropy terms, which
is what prevents the regression term from dominating the combined loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

import config


@dataclass
class LossOutputs:
    total_loss: torch.Tensor
    regression_loss: torch.Tensor
    fade_risk_loss: torch.Tensor
    maintenance_loss: torch.Tensor


class MultiTaskLoss(nn.Module):
    """Weighted combination of Huber regression loss + two cross-entropy losses."""

    def __init__(self, train_config: config.TrainConfig):
        super().__init__()
        self.regression_loss_fn = nn.SmoothL1Loss(beta=0.05)
        self.fade_risk_loss_fn = nn.CrossEntropyLoss()
        self.maintenance_loss_fn = nn.CrossEntropyLoss()

        self.w_regression = train_config.loss_weight_regression
        self.w_fade_risk = train_config.loss_weight_fade_risk
        self.w_maintenance = train_config.loss_weight_maintenance

    def forward(
        self,
        regression_pred: torch.Tensor,
        fade_risk_logits: torch.Tensor,
        maintenance_logits: torch.Tensor,
        regression_target: torch.Tensor,
        fade_risk_target: torch.Tensor,
        maintenance_target: torch.Tensor,
    ) -> LossOutputs:
        regression_loss = self.regression_loss_fn(regression_pred, regression_target)
        fade_risk_loss = self.fade_risk_loss_fn(fade_risk_logits, fade_risk_target)
        maintenance_loss = self.maintenance_loss_fn(maintenance_logits, maintenance_target)

        total_loss = (
            self.w_regression * regression_loss
            + self.w_fade_risk * fade_risk_loss
            + self.w_maintenance * maintenance_loss
        )

        return LossOutputs(
            total_loss=total_loss,
            regression_loss=regression_loss,
            fade_risk_loss=fade_risk_loss,
            maintenance_loss=maintenance_loss,
        )
