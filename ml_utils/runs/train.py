"""PyTorch Lightning training infrastructure."""

from pathlib import Path
from typing import Any, Literal

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.strategies import Strategy

from ml_utils.data import TensorDataModule, stratified_split
from ml_utils.nets.base import MlBaseModule
from ml_utils.nets.classifier import MlClassifier
from ml_utils.runs.options import Options
from ml_utils.utils.metrics import ClassificationResult

# Simplified names (primary API)
BaseModule = MlBaseModule
Classifier = MlClassifier
DataModule = TensorDataModule


def create_trainer(
    project: str,
    run_name: str | None = None,
    options: Options | None = None,
    max_epochs: int = 100,
    early_stopping: bool = True,
    patience: int = 10,
    checkpoint_dir: str | Path | None = None,
    enable_progress_bar: bool = True,
    gradient_clip_val: float | None = None,
    gradient_clip_algorithm: str | None = None,
    accumulate_grad_batches: int = 1,
    strategy: str | Strategy = "auto",
    devices: int | list[int] | str = "auto",
    num_nodes: int = 1,
    watch_model: bool = True,
    watch_log: Literal["gradients", "parameters", "all"] = "gradients",
    watch_freq: int = 100,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
    offline: bool = False,
    logger: Logger | None | bool = True,
    **kwargs: Any,
) -> pl.Trainer:
    """Create a configured Trainer with W&B logging.

    Args:
        project: W&B project name (required)
        run_name: W&B run name (auto-generated if None)
        options: Execution options (precision, compile mode)
        max_epochs: Maximum training epochs
        early_stopping: Enable early stopping callback
        patience: Early stopping patience
        checkpoint_dir: Directory for model checkpoints
        enable_progress_bar: Show progress bar
        gradient_clip_val: Max gradient norm for clipping
        gradient_clip_algorithm: Gradient clipping algorithm ("value" or "norm")
        accumulate_grad_batches: Gradient accumulation steps
        strategy: Distributed strategy ("ddp", "fsdp", "auto", or Strategy instance)
        devices: Number of devices or list of device indices
        num_nodes: Number of nodes for multi-node training
        watch_model: Enable wandb.watch() for gradient/parameter monitoring
        watch_log: What to log - 'gradients', 'parameters', or 'all'
        watch_freq: Log frequency for gradients/parameters
        tags: W&B run tags
        config: Hyperparameters to log to W&B
        offline: Run W&B in offline mode
        logger: Logger to use. True creates a W&B logger, False disables
            logging, or pass a custom logger instance.
        **kwargs: Additional Trainer kwargs

    Returns:
        Configured pl.Trainer with W&B logger
    """
    import wandb as wandb_module

    from ml_utils.utils.track import create_wandb_logger as create_logger

    options = options or Options()
    trainer_kwargs = options.trainer_kwargs()

    # Handle logger configuration
    if logger is True:
        trainer_logger: Logger | None = create_logger(
            project=project,
            name=run_name,
            tags=tags,
            config=config,
            save_dir=str(checkpoint_dir) if checkpoint_dir else ".",
            offline=offline,
        )
    elif logger is False:
        trainer_logger = None
    else:
        trainer_logger = logger

    callbacks: list[Callback] = []

    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=patience,
                mode="min",
            )
        )

    if checkpoint_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best-{epoch}-{val/loss:.4f}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        )

    # Only add LearningRateMonitor if we have a logger
    if trainer_logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Add wandb.watch() callback for gradient/parameter monitoring
    if watch_model and trainer_logger is not None:

        class WatchCallback(Callback):
            """Callback to enable wandb.watch() at training start."""

            def on_train_start(
                self, trainer: pl.Trainer, pl_module: pl.LightningModule
            ) -> None:
                if wandb_module.run is not None:
                    wandb_module.watch(
                        pl_module,
                        log=watch_log,
                        log_freq=watch_freq,
                        log_graph=True,
                    )

        callbacks.append(WatchCallback())

    return pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=trainer_logger,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        accumulate_grad_batches=accumulate_grad_batches,
        strategy=strategy,
        devices=devices,
        num_nodes=num_nodes,
        **trainer_kwargs,
        **kwargs,
    )


__all__ = [
    "BaseModule",
    "Classifier",
    "ClassificationResult",
    "DataModule",
    "create_trainer",
    "stratified_split",
]
