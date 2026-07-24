"""
metrics.py
==========

Evaluation metrics for the regression head (Brake Health %) and the two
classification heads (Fade Risk, Maintenance Recommendation).

Classification reports are always generated with an explicit `labels=`
argument covering the FULL fixed class set (config.FADE_RISK_CLASSES /
config.MAINTENANCE_CLASSES), not just the classes observed in a given
batch or split. This is what prevents the
"sklearn classification_report class mismatch when test splits lack all
classes" bug encountered in the previous session: without an explicit
`labels=` list, scikit-learn infers labels from the data present, and a
split that happens to omit a rare class (e.g. "Emergency Stop") produces
a report with a different shape than expected, or raises when compared
against a fixed-shape target array.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import config


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    report_text: str
    confusion: np.ndarray


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute MAE / RMSE / R2 on DE-SCALED (real-world, 0-100%) values.

    Callers are responsible for inverse-transforming the MinMax-scaled
    model outputs before calling this function, so these metrics are
    always reported in interpretable percentage-point units.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    return RegressionMetrics(mae=mae, rmse=rmse, r2=r2)


def _all_class_indices(num_classes: int) -> List[int]:
    return list(range(num_classes))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> ClassificationMetrics:
    """Compute accuracy, macro/weighted F1, a full text report, and a
    confusion matrix -- all computed against the FULL fixed label set
    (0..len(class_names)-1) regardless of which classes actually appear
    in `y_true`/`y_pred`. This guarantees stable, fixed-shape output even
    on small or imbalanced evaluation splits.
    """
    labels = _all_class_indices(len(class_names))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )

    confusion = confusion_matrix(y_true, y_pred, labels=labels)

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        report_text=report_text,
        confusion=confusion,
    )


def format_confusion_matrix(confusion: np.ndarray, class_names: List[str]) -> str:
    """Pretty-print a confusion matrix as an aligned text table."""
    col_width = max(len(name) for name in class_names) + 2
    header = " " * col_width + "".join(f"{name[:10]:>12}" for name in class_names)
    lines = [header]
    for i, name in enumerate(class_names):
        row = f"{name:<{col_width}}" + "".join(f"{confusion[i, j]:>12}" for j in range(len(class_names)))
        lines.append(row)
    return "\n".join(lines)
