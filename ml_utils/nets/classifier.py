"""Classification module with metrics and evaluation.

Provides MlClassifier with typed metrics via model.metrics.{metric}.calc().
"""

import torch
import torchmetrics
from torch import Tensor, nn

from ml_utils.nets.base import MlBaseModule
from ml_utils.runs.options import Options
from ml_utils.utils.metrics import ClassificationResult, ClassifierMetrics


class MlClassifier(MlBaseModule):
    _num_classes: int
    _class_names: tuple[str, ...]
    metrics: ClassifierMetrics

    def __init__(
        self,
        num_classes: int,
        class_names: tuple[str, ...],
        seed: int = 42,
        options: Options | None = None,
        lr: float = 0.01,
        class_weights: Tensor | None = None,
    ) -> None:
        if len(class_names) != num_classes:
            msg = f"class_names length {len(class_names)} != num_classes {num_classes}"
            raise ValueError(msg)

        self._num_classes = num_classes
        self._class_names = class_names
        super().__init__(seed, options, lr, class_weights)

        self.metrics = ClassifierMetrics(self, num_classes, class_names)

        self._train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self._val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self._val_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self._val_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self._val_confusion = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return self._num_classes

    @property
    def class_names(self) -> tuple[str, ...]:
        """Human-readable class names."""
        return self._class_names

    def _configure_loss(self, class_weights: Tensor | None = None) -> nn.Module:
        """Default to CrossEntropyLoss for classification."""
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights.float())
        return nn.CrossEntropyLoss()

    @property
    def _loss(self) -> nn.Module:
        if self._criterion is None:
            raise RuntimeError("Training requires a configured learning rate")
        return self._criterion

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step with classification metrics."""
        inputs, targets = batch
        features = self._extract_features(inputs)
        logits = self._forward(features)
        loss = self._loss(logits, targets)

        preds = logits.argmax(dim=-1)
        self._train_acc(preds, targets)
        self._train_f1(preds, targets)

        batch_size = targets.size(0)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/acc",
            self._train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train/f1",
            self._train_f1,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Validation step with classification metrics."""
        inputs, targets = batch
        features = self._extract_features(inputs)
        logits = self._forward(features)
        loss = self._loss(logits, targets)

        preds = logits.argmax(dim=-1)
        self._val_acc(preds, targets)
        self._val_f1(preds, targets)
        self._val_precision(preds, targets)
        self._val_recall(preds, targets)
        self._val_confusion(preds, targets)

        batch_size = targets.size(0)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val/acc",
            self._val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val/f1",
            self._val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val/precision",
            self._val_precision,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/recall",
            self._val_recall,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Test step with classification metrics."""
        inputs, targets = batch
        features = self._extract_features(inputs)
        logits = self._forward(features)
        loss = self._loss(logits, targets)

        preds = logits.argmax(dim=-1)
        self._val_acc(preds, targets)
        self._val_f1(preds, targets)

        batch_size = targets.size(0)
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(
            "test/acc",
            self._val_acc,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "test/f1", self._val_f1, on_step=False, on_epoch=True, batch_size=batch_size
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log confusion matrix to W&B at end of validation epoch."""
        confusion = self._val_confusion.compute()
        self.metrics.confusion._log_confusion("val/confusion_matrix", confusion)
        self._val_confusion.reset()

    def predict_step(
        self, batch: Tensor | tuple[Tensor, ...], batch_idx: int
    ) -> Tensor:
        """Batch prediction step."""
        if isinstance(batch, tuple):
            inputs = batch[0]
        else:
            inputs = batch
        features = self._extract_features(inputs)
        logits = self._forward(features)
        return logits.argmax(dim=-1)

    def predict(self, inputs: Tensor) -> Tensor:
        """Predict class labels for inputs.

        Args:
            inputs: Input tensor [N, ...]

        Returns:
            Predicted class labels [N] as long tensor
        """
        self.eval()
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            features = self._extract_features(inputs)
            logits = self._forward(features)
            preds = logits.argmax(dim=-1)
        return preds

    def predict_logits(self, inputs: Tensor) -> Tensor:
        """Get raw logits for inputs.

        Args:
            inputs: Input tensor [N, ...]

        Returns:
            Logits tensor [N, C]
        """
        self.eval()
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            features = self._extract_features(inputs)
            return self._forward(features)

    def evaluate(
        self,
        inputs: Tensor,
        targets: Tensor,
        *,
        log: bool = True,
        key_prefix: str = "eval",
    ) -> ClassificationResult:
        """Evaluate classifier on labeled data.

        Args:
            inputs: Input tensor [N, ...]
            targets: True class labels [N] as long tensor
            log: Whether to log to W&B
            key_prefix: Prefix for W&B keys

        Returns:
            ClassificationResult with all metrics
        """
        preds = self.predict(inputs)
        targets = targets.to(preds.device)
        return self.metrics.calc(preds, targets, log=log, key_prefix=key_prefix)

    def calc_misclassifications(
        self,
        inputs: Tensor,
        targets: Tensor,
        num_samples: int = 50,
    ) -> dict[str, float]:
        """Analyze misclassified samples. Logs to W&B.

        Args:
            inputs: Input features [N, ...]
            targets: True class labels [N]
            num_samples: Number of misclassifications to log

        Returns:
            Dict with misclassification statistics
        """
        import wandb

        predictions = self.predict(inputs)
        targets = targets.to(predictions.device)

        incorrect_mask = predictions != targets
        incorrect_indices = torch.where(incorrect_mask)[0]
        error_rate = len(incorrect_indices) / len(predictions)

        stats = {
            "misclassification_rate": error_rate,
            "total_errors": len(incorrect_indices),
            "total_samples": len(predictions),
        }

        if wandb.run is None:
            return stats

        if len(incorrect_indices) == 0:
            wandb.log({"analysis/misclassification_rate": 0.0})
            return stats

        sample_size = min(num_samples, len(incorrect_indices))
        sampled_indices = incorrect_indices[
            torch.randperm(len(incorrect_indices))[:sample_size]
        ]

        table_data = []
        confusion_pairs: dict[tuple[str, str], int] = {}

        for idx in sampled_indices:
            true_label = self._class_names[int(targets[idx].item())]
            pred_label = self._class_names[int(predictions[idx].item())]

            pair = (true_label, pred_label)
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

            features = inputs[idx].cpu().numpy().flatten()[:10].tolist()
            table_data.append([int(idx), true_label, pred_label, features])

        table = wandb.Table(
            columns=["sample_id", "true_class", "predicted_class", "features_preview"],
            data=table_data,
        )
        wandb.log({"analysis/misclassifications": table})
        wandb.log({"analysis/misclassification_rate": error_rate})

        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        confusion_data = [
            [f"{true_cls} -> {pred_cls}", count]
            for (true_cls, pred_cls), count in sorted_pairs[:10]
        ]

        if confusion_data:
            confusion_table = wandb.Table(
                columns=["confusion_pair", "count"],
                data=confusion_data,
            )
            wandb.log({"analysis/top_confusion_pairs": confusion_table})

        return stats

    def calc_confidence(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> dict[str, float]:
        """Analyze prediction confidence and calibration. Logs to W&B.

        Args:
            inputs: Input features [N, ...]
            targets: True class labels [N]

        Returns:
            Dict with confidence statistics
        """
        import wandb

        logits = self.predict_logits(inputs)
        predictions = logits.argmax(dim=-1)
        targets = targets.to(predictions.device)

        probs = torch.softmax(logits, dim=-1)
        pred_probs = probs.gather(1, predictions.unsqueeze(1)).squeeze(1)

        correct_mask = predictions == targets
        correct_probs = pred_probs[correct_mask]
        incorrect_probs = pred_probs[~correct_mask]

        stats: dict[str, float] = {}

        if len(correct_probs) > 0:
            stats["confidence_correct_mean"] = correct_probs.mean().item()
            stats["confidence_correct_std"] = correct_probs.std().item()

        if len(incorrect_probs) > 0:
            stats["confidence_incorrect_mean"] = incorrect_probs.mean().item()
            stats["confidence_incorrect_std"] = incorrect_probs.std().item()

        if wandb.run is None:
            return stats

        if len(correct_probs) > 0:
            wandb.log(
                {
                    "analysis/confidence_correct_mean": stats[
                        "confidence_correct_mean"
                    ],
                    "analysis/confidence_correct_std": stats["confidence_correct_std"],
                    "analysis/confidence_correct_hist": wandb.Histogram(
                        correct_probs.cpu().tolist()
                    ),
                }
            )

        if len(incorrect_probs) > 0:
            wandb.log(
                {
                    "analysis/confidence_incorrect_mean": stats[
                        "confidence_incorrect_mean"
                    ],
                    "analysis/confidence_incorrect_std": stats[
                        "confidence_incorrect_std"
                    ],
                    "analysis/confidence_incorrect_hist": wandb.Histogram(
                        incorrect_probs.cpu().tolist()
                    ),
                }
            )

        return stats

    def log_samples(
        self,
        inputs: Tensor,
        targets: Tensor,
        num_samples: int = 20,
        key: str = "predictions/samples",
    ) -> None:
        """Log sample predictions to W&B as a table.

        Args:
            inputs: Input features [N, ...]
            targets: True class labels [N]
            num_samples: Number of samples to log
            key: W&B log key
        """
        import wandb

        if wandb.run is None:
            return

        predictions = self.predict(inputs)
        targets = targets.to(predictions.device)

        correct_mask = predictions == targets
        correct_indices = torch.where(correct_mask)[0]
        incorrect_indices = torch.where(~correct_mask)[0]

        n_correct = min(num_samples // 2, len(correct_indices))
        n_incorrect = min(num_samples - n_correct, len(incorrect_indices))

        if n_correct + n_incorrect == 0:
            return

        sample_indices = torch.cat(
            [
                correct_indices[:n_correct],
                incorrect_indices[:n_incorrect],
            ]
        )

        table_data = []
        for idx in sample_indices:
            pred_label = self._class_names[int(predictions[idx].item())]
            true_label = self._class_names[int(targets[idx].item())]
            correct = predictions[idx] == targets[idx]
            features = inputs[idx].cpu().numpy().flatten()[:5].tolist()

            table_data.append(
                [
                    int(idx),
                    true_label,
                    pred_label,
                    correct.item(),
                    features,
                ]
            )

        table = wandb.Table(
            columns=[
                "sample_id",
                "true_label",
                "predicted_label",
                "correct",
                "features_preview",
            ],
            data=table_data,
        )

        wandb.log({key: table})
