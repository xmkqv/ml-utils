"""Unified metrics system fusing torchmetrics with W&B logging.

Each metric class wraps torchmetrics computation with optional W&B logging.
Access via model.metrics.{metric}.calc(preds, targets, log=True).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

if TYPE_CHECKING:
    from ml_utils.nets.base import MlBaseModule


class Metric[T](ABC):
    """Base metric with calc and log capabilities."""

    def __init__(self, model: "MlBaseModule") -> None:
        self._model = model

    @abstractmethod
    def calc(self, preds: Tensor, targets: Tensor, *, log: bool = True) -> T:
        """Calculate metric, optionally log to W&B."""
        ...

    def _log(self, key: str, value: float | Tensor) -> None:
        """Log scalar or tensor to W&B via model's logger."""
        import wandb

        if wandb.run is None:
            return

        logged = (
            value.item() if isinstance(value, Tensor) and value.numel() == 1 else value
        )
        wandb.log({key: cast(Any, logged)})

    def _log_table(self, key: str, columns: list[str], data: list[list]) -> None:
        """Log table to W&B."""
        import wandb

        if wandb.run is None:
            return
        wandb.log({key: wandb.Table(columns=columns, data=data)})

    @property
    def device(self) -> torch.device:
        """Model device."""
        return self._model.device


class Accuracy(Metric[float]):
    """Multiclass accuracy metric."""

    def __init__(self, model: "MlBaseModule", num_classes: int) -> None:
        super().__init__(model)
        self._num_classes = num_classes

    def calc(
        self,
        preds: Tensor,
        targets: Tensor,
        *,
        log: bool = True,
        key: str = "accuracy",
    ) -> float:
        metric = MulticlassAccuracy(self._num_classes, average="micro").to(self.device)
        result = metric(preds, targets).item()
        if log:
            self._log(key, result)
        return result


class F1(Metric[float]):
    """Multiclass F1 score (macro average)."""

    def __init__(self, model: "MlBaseModule", num_classes: int) -> None:
        super().__init__(model)
        self._num_classes = num_classes

    def calc(
        self,
        preds: Tensor,
        targets: Tensor,
        *,
        log: bool = True,
        key: str = "f1_macro",
    ) -> float:
        metric = MulticlassF1Score(self._num_classes, average="macro").to(self.device)
        result = metric(preds, targets).item()
        if log:
            self._log(key, result)
        return result


@dataclass(frozen=True, slots=True)
class F1PerClassResult:
    """Per-class F1 scores."""

    scores: tuple[float, ...]
    class_names: tuple[str, ...] | None


class F1PerClass(Metric[F1PerClassResult]):
    """Per-class F1 scores."""

    def __init__(
        self,
        model: "MlBaseModule",
        num_classes: int,
        class_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(model)
        self._num_classes = num_classes
        self._class_names = class_names

    def calc(
        self,
        preds: Tensor,
        targets: Tensor,
        *,
        log: bool = True,
        key: str = "f1_per_class",
    ) -> F1PerClassResult:
        metric = MulticlassF1Score(self._num_classes, average="none").to(self.device)
        scores = tuple(metric(preds, targets).tolist())
        result = F1PerClassResult(scores=scores, class_names=self._class_names)

        if log and self._class_names is not None:
            data = [[name, score] for name, score in zip(self._class_names, scores)]
            self._log_table(key, ["class", "f1"], data)

        return result


class Precision(Metric[float]):
    """Multiclass precision (macro average)."""

    def __init__(self, model: "MlBaseModule", num_classes: int) -> None:
        super().__init__(model)
        self._num_classes = num_classes

    def calc(
        self,
        preds: Tensor,
        targets: Tensor,
        *,
        log: bool = True,
        key: str = "precision",
    ) -> float:
        metric = MulticlassPrecision(self._num_classes, average="macro").to(self.device)
        result = metric(preds, targets).item()
        if log:
            self._log(key, result)
        return result


@dataclass(frozen=True, slots=True)
class PrecisionPerClassResult:
    """Per-class precision scores."""

    scores: tuple[float, ...]
    class_names: tuple[str, ...] | None


class PrecisionPerClass(Metric[PrecisionPerClassResult]):
    """Per-class precision scores."""

    def __init__(
        self,
        model: "MlBaseModule",
        num_classes: int,
        class_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(model)
        self._num_classes = num_classes
        self._class_names = class_names

    def calc(
        self,
        preds: Tensor,
        targets: Tensor,
        *,
        log: bool = True,
        key: str = "precision_per_class",
    ) -> PrecisionPerClassResult:
        metric = MulticlassPrecision(self._num_classes, average="none").to(self.device)
        scores = tuple(metric(preds, targets).tolist())
        result = PrecisionPerClassResult(scores=scores, class_names=self._class_names)

        if log and self._class_names is not None:
            data = [[name, score] for name, score in zip(self._class_names, scores)]
            self._log_table(key, ["class", "precision"], data)

        return result


class Recall(Metric[float]):
    """Multiclass recall (macro average)."""

    def __init__(self, model: "MlBaseModule", num_classes: int) -> None:
        super().__init__(model)
        self._num_classes = num_classes

    def calc(
        self,
        preds: Tensor,
        targets: Tensor,
        *,
        log: bool = True,
        key: str = "recall",
    ) -> float:
        metric = MulticlassRecall(self._num_classes, average="macro").to(self.device)
        result = metric(preds, targets).item()
        if log:
            self._log(key, result)
        return result


@dataclass(frozen=True, slots=True)
class RecallPerClassResult:
    """Per-class recall scores."""

    scores: tuple[float, ...]
    class_names: tuple[str, ...] | None


class RecallPerClass(Metric[RecallPerClassResult]):
    """Per-class recall scores."""

    def __init__(
        self,
        model: "MlBaseModule",
        num_classes: int,
        class_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(model)
        self._num_classes = num_classes
        self._class_names = class_names

    def calc(
        self,
        preds: Tensor,
        targets: Tensor,
        *,
        log: bool = True,
        key: str = "recall_per_class",
    ) -> RecallPerClassResult:
        metric = MulticlassRecall(self._num_classes, average="none").to(self.device)
        scores = tuple(metric(preds, targets).tolist())
        result = RecallPerClassResult(scores=scores, class_names=self._class_names)

        if log and self._class_names is not None:
            data = [[name, score] for name, score in zip(self._class_names, scores)]
            self._log_table(key, ["class", "recall"], data)

        return result


class ConfusionMatrix(Metric[Tensor]):
    """Multiclass confusion matrix."""

    def __init__(
        self,
        model: "MlBaseModule",
        num_classes: int,
        class_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(model)
        self._num_classes = num_classes
        self._class_names = class_names

    def calc(
        self,
        preds: Tensor,
        targets: Tensor,
        *,
        log: bool = True,
        key: str = "confusion_matrix",
    ) -> Tensor:
        metric = MulticlassConfusionMatrix(self._num_classes).to(self.device)
        result = metric(preds, targets)

        if log:
            self._log_confusion(key, result)

        return result

    def _log_confusion(self, key: str, matrix: Tensor) -> None:
        """Log confusion matrix to W&B."""
        import wandb

        if wandb.run is None:
            return

        cm_data = matrix.cpu().tolist()
        names = (
            list(self._class_names)
            if self._class_names
            else [str(i) for i in range(self._num_classes)]
        )

        wandb.log({key: wandb.Table(data=cm_data, columns=names)})


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Complete classification metrics."""

    accuracy: float
    f1_macro: float
    f1_per_class: tuple[float, ...]
    precision_macro: float
    precision_per_class: tuple[float, ...]
    recall_macro: float
    recall_per_class: tuple[float, ...]
    confusion: Tensor
    class_names: tuple[str, ...] | None


def _calc_all_classification(
    model: "MlBaseModule",
    preds: Tensor,
    targets: Tensor,
    num_classes: int,
    class_names: tuple[str, ...] | None,
    *,
    log: bool = True,
    key_prefix: str = "eval",
) -> ClassificationResult:
    """Compute all classification metrics at once."""
    device = model.device

    accuracy = MulticlassAccuracy(num_classes, average="micro").to(device)
    f1_macro = MulticlassF1Score(num_classes, average="macro").to(device)
    f1_none = MulticlassF1Score(num_classes, average="none").to(device)
    prec_macro = MulticlassPrecision(num_classes, average="macro").to(device)
    prec_none = MulticlassPrecision(num_classes, average="none").to(device)
    rec_macro = MulticlassRecall(num_classes, average="macro").to(device)
    rec_none = MulticlassRecall(num_classes, average="none").to(device)
    confusion = MulticlassConfusionMatrix(num_classes).to(device)

    result = ClassificationResult(
        accuracy=accuracy(preds, targets).item(),
        f1_macro=f1_macro(preds, targets).item(),
        f1_per_class=tuple(f1_none(preds, targets).tolist()),
        precision_macro=prec_macro(preds, targets).item(),
        precision_per_class=tuple(prec_none(preds, targets).tolist()),
        recall_macro=rec_macro(preds, targets).item(),
        recall_per_class=tuple(rec_none(preds, targets).tolist()),
        confusion=confusion(preds, targets),
        class_names=class_names,
    )

    if log:
        _log_classification(result, num_classes, key_prefix)

    return result


def _log_classification(
    result: ClassificationResult, num_classes: int, prefix: str
) -> None:
    """Log all classification metrics to W&B."""
    import wandb

    if wandb.run is None:
        return

    wandb.log(
        {
            f"{prefix}/accuracy": result.accuracy,
            f"{prefix}/f1_macro": result.f1_macro,
            f"{prefix}/precision_macro": result.precision_macro,
            f"{prefix}/recall_macro": result.recall_macro,
        }
    )

    if result.class_names is not None:
        data = [
            [name, f1, prec, rec]
            for name, f1, prec, rec in zip(
                result.class_names,
                result.f1_per_class,
                result.precision_per_class,
                result.recall_per_class,
            )
        ]
        table = wandb.Table(
            columns=["class", "f1", "precision", "recall"],
            data=data,
        )
        wandb.log({f"{prefix}/per_class": table})

        try:
            wandb.log(
                {
                    f"{prefix}/f1_by_class": wandb.plot.bar(
                        table, "class", "f1", title="F1 by Class"
                    )
                }
            )
        except Exception:
            pass

    cm_data = result.confusion.cpu().tolist()
    names = (
        list(result.class_names)
        if result.class_names
        else [str(i) for i in range(num_classes)]
    )

    wandb.log({f"{prefix}/confusion": wandb.Table(data=cm_data, columns=names)})


class Metrics:
    """Base metrics container. Extend for domain-specific metrics."""

    def __init__(self, model: "MlBaseModule") -> None:
        self._model = model


class ClassifierMetrics(Metrics):
    """Metrics container for classification models.

    Usage:
        model.metrics.calc(preds, targets)  # all metrics
        model.metrics.accuracy.calc(preds, targets)  # single metric
    """

    accuracy: Accuracy
    f1: F1
    f1_per_class: F1PerClass
    precision: Precision
    precision_per_class: PrecisionPerClass
    recall: Recall
    recall_per_class: RecallPerClass
    confusion: ConfusionMatrix

    def __init__(
        self,
        model: "MlBaseModule",
        num_classes: int,
        class_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(model)
        self._num_classes = num_classes
        self._class_names = class_names
        self.accuracy = Accuracy(model, num_classes)
        self.f1 = F1(model, num_classes)
        self.f1_per_class = F1PerClass(model, num_classes, class_names)
        self.precision = Precision(model, num_classes)
        self.precision_per_class = PrecisionPerClass(model, num_classes, class_names)
        self.recall = Recall(model, num_classes)
        self.recall_per_class = RecallPerClass(model, num_classes, class_names)
        self.confusion = ConfusionMatrix(model, num_classes, class_names)

    def calc(
        self,
        preds: Tensor,
        targets: Tensor,
        *,
        log: bool = True,
        key_prefix: str = "eval",
    ) -> ClassificationResult:
        """Compute all classification metrics."""
        return _calc_all_classification(
            self._model,
            preds,
            targets,
            self._num_classes,
            self._class_names,
            log=log,
            key_prefix=key_prefix,
        )
