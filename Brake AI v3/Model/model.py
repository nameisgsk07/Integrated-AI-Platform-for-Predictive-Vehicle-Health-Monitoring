"""
model.py
========

Multi-task neural network architecture for EdgeGuard AI.

Architecture:
    Shared backbone (Residual Fully-Connected blocks + BatchNorm + Dropout)
      -> Regression head    (Brake Health %, 1 output, sigmoid-bounded to [0, 1]
                              since the target is MinMax-scaled to [0, 1])
      -> Classification head A (Brake Fade Risk, 5 classes)
      -> Classification head B (Maintenance Recommendation, 7 classes)

All linear layers use Kaiming initialization (appropriate for the ReLU
family of activations used throughout).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

import config


def _kaiming_init(module: nn.Module) -> None:
    """Apply Kaiming (He) initialization to Linear layers, zero-init biases."""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ResidualFCBlock(nn.Module):
    """A residual fully-connected block:

        x -> Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> BatchNorm
          -> (+ skip connection, projected if dimensions differ)
          -> ReLU

    This lets gradients flow directly through the skip path, which
    stabilizes training of the deeper shared backbone.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout_p: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.activation = nn.ReLU(inplace=True)

        self.needs_projection = in_dim != out_dim
        if self.needs_projection:
            self.projection = nn.Linear(in_dim, out_dim)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.bn2(out)

        if self.projection is not None:
            identity = self.projection(identity)

        out = out + identity
        out = self.activation(out)
        return out


class SharedBackbone(nn.Module):
    """Stack of Residual FC blocks shared across all three prediction heads."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_p: float):
        super().__init__()
        dims = [input_dim] + hidden_dims
        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(ResidualFCBlock(dims[i], dims[i + 1], dropout_p))
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class PredictionHead(nn.Module):
    """A small MLP head mapping shared-backbone features to task-specific outputs."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeGuardNet(nn.Module):
    """Full multi-task network: shared backbone + 1 regression head + 2 classification heads."""

    def __init__(self, model_config: config.ModelConfig):
        super().__init__()
        self.backbone = SharedBackbone(
            input_dim=model_config.input_dim,
            hidden_dims=model_config.backbone_hidden_dims,
            dropout_p=model_config.dropout_p,
        )

        backbone_out = self.backbone.output_dim

        self.regression_head = PredictionHead(
            input_dim=backbone_out,
            hidden_dim=model_config.head_hidden_dim,
            output_dim=1,
            dropout_p=model_config.dropout_p,
        )
        self.fade_risk_head = PredictionHead(
            input_dim=backbone_out,
            hidden_dim=model_config.head_hidden_dim,
            output_dim=model_config.num_fade_risk_classes,
            dropout_p=model_config.dropout_p,
        )
        self.maintenance_head = PredictionHead(
            input_dim=backbone_out,
            hidden_dim=model_config.head_hidden_dim,
            output_dim=model_config.num_maintenance_classes,
            dropout_p=model_config.dropout_p,
        )

        self.apply(_kaiming_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (regression_output, fade_risk_logits, maintenance_logits).

        `regression_output` is passed through a sigmoid so it is bounded to
        [0, 1], matching the MinMax-scaled regression target. It is
        de-scaled back to a 0-100% Brake Health value downstream (in
        predict.py) using the fitted regression scaler.
        """
        features = self.backbone(x)
        regression_raw = self.regression_head(features)
        regression_output = torch.sigmoid(regression_raw)
        fade_risk_logits = self.fade_risk_head(features)
        maintenance_logits = self.maintenance_head(features)
        return regression_output, fade_risk_logits, maintenance_logits

    def enable_mc_dropout(self) -> None:
        """Put the model in eval mode EXCEPT keep Dropout layers active.

        This is required for Monte Carlo Dropout confidence estimation:
        BatchNorm layers must use running statistics (eval mode), but
        Dropout layers must remain stochastic (train mode) so repeated
        forward passes produce a distribution of outputs.
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


def build_model(model_config: config.ModelConfig) -> EdgeGuardNet:
    """Factory function used by both train.py and predict.py."""
    return EdgeGuardNet(model_config)
