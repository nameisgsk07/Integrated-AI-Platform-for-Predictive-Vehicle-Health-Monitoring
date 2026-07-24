"""
model.py
========
Neural network architecture for the EdgeGuard AI Brake Health Prediction
model: a shared feature-extraction backbone feeding four independent output
heads (two regression, two classification).

Designed to be lightweight and ONNX/TensorRT friendly (only Linear,
BatchNorm1d, ReLU and Dropout layers are used) so it can later be exported
for deployment on an automotive infotainment ECU.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from config import EdgeGuardConfig, ModelConfig, OutputConstraints


class SharedBackbone(nn.Module):
    """Stack of Linear -> BatchNorm -> ReLU -> Dropout blocks that learns
    general brake-behaviour representations shared by every head."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.blocks = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class RegressionHead(nn.Module):
    """Private hidden layer + single-value output projection.

    Regression heads never share their final layers with one another or
    with the backbone beyond the shared feature vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return self.output_layer(h).squeeze(-1)


class ClassificationHead(nn.Module):
    """Private hidden layer + independent class-logit projection."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return self.output_layer(h)


class EdgeGuardBrakeNet(nn.Module):
    """Multi-task network: one shared backbone, four independent heads.

    Outputs (raw, un-clamped):
        brake_health         -> scalar regression (should sit near 0-100)
        remaining_pad_life   -> scalar regression (should sit near 0-50000)
        fade_risk_logits     -> classification logits
        maintenance_logits   -> classification logits
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.backbone = SharedBackbone(
            input_dim=model_config.input_dim,
            hidden_dims=model_config.backbone_hidden_dims,
            dropout=model_config.dropout,
        )
        feat_dim = self.backbone.output_dim

        self.brake_health_head = RegressionHead(feat_dim, model_config.head_hidden_dim, model_config.dropout)
        self.remaining_pad_life_head = RegressionHead(feat_dim, model_config.head_hidden_dim, model_config.dropout)
        self.fade_risk_head = ClassificationHead(
            feat_dim, model_config.head_hidden_dim, model_config.num_fade_risk_classes, model_config.dropout
        )
        self.maintenance_action_head = ClassificationHead(
            feat_dim, model_config.head_hidden_dim, model_config.num_maintenance_classes, model_config.dropout
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            "brake_health": self.brake_health_head(features),
            "remaining_pad_life": self.remaining_pad_life_head(features),
            "fade_risk_logits": self.fade_risk_head(features),
            "maintenance_logits": self.maintenance_action_head(features),
        }

    def enable_mc_dropout(self) -> None:
        """Puts every Dropout submodule into training mode while leaving
        BatchNorm (and everything else) in eval mode. Used to perform Monte
        Carlo Dropout sampling for regression confidence estimation at
        inference time without disturbing BatchNorm running statistics."""
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


def build_model(config: EdgeGuardConfig) -> EdgeGuardBrakeNet:
    return EdgeGuardBrakeNet(config.model)


def clamp_outputs(brake_health: torch.Tensor, remaining_pad_life: torch.Tensor,
                   constraints: OutputConstraints):
    """Clamps regression outputs to physically valid ranges. Applied at
    inference / reporting time so that raw training gradients are never
    distorted by clamping."""
    brake_health = torch.clamp(brake_health, constraints.brake_health_min, constraints.brake_health_max)
    remaining_pad_life = torch.clamp(
        remaining_pad_life, constraints.remaining_pad_life_min, constraints.remaining_pad_life_max
    )
    return brake_health, remaining_pad_life
