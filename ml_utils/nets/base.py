"""ML module with profiling, options, and training support.

Provides MlBaseModule - unified foundation for all ML models.
"""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn

from ml_utils.runs.options import CompileMode, Options


class MlBaseModule(pl.LightningModule, ABC):
    """ML module with profiling, options, and training support.

    All interfaces use pure torch.Tensor - no numpy.
    All logging goes through W&B via Lightning's logger integration.

    Provides:
    - Options integration for Trainer configuration
    - Profiling utilities (benchmark_latency, profile_memory)
    - Serialization helpers
    - Training step with feature extraction
    - Validation/test steps
    - Optimizer/scheduler configuration

    Subclasses must implement:
    - _build_net(): Construct the neural network

    Optional overrides:
    - _extract_features(): Transform inputs to features (default: passthrough)
    - _forward(): Forward pass on features (default: net(features))
    - _configure_loss(): Return loss function (default: MSELoss)
    - _configure_optimizer(): Return optimizer (default: Adam)
    - _configure_scheduler(): Return LR scheduler (default: None)
    """

    _options: Options
    _net: nn.Module
    _lr: float | None
    _criterion: nn.Module | None

    def __init__(
        self,
        seed: int = 42,
        options: Options | None = None,
        lr: float | None = None,
        class_weights: Tensor | None = None,
    ) -> None:
        super().__init__()
        pl.seed_everything(seed, workers=True)
        self._options = options or Options()
        self._net = self._build_net()
        self._lr = lr
        self._criterion = (
            self._configure_loss(class_weights) if lr is not None else None
        )

        if lr is not None:
            self.save_hyperparameters(ignore=["options", "class_weights"])

    @abstractmethod
    def _build_net(self) -> nn.Module:
        """Construct the neural network."""
        ...

    def _extract_features(self, inputs: Tensor) -> Tensor:
        """Transform raw inputs to feature tensors. Override for custom extraction."""
        return inputs

    def _configure_loss(self, class_weights: Tensor | None = None) -> nn.Module:
        """Return loss function. Override for custom loss."""
        return nn.MSELoss()

    def _configure_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Return optimizer. Override for custom optimizer."""
        return torch.optim.Adam(
            self.parameters(),
            lr=lr,
            foreach=True,
            capturable=self._options.cuda_graphs,
        )

    def _configure_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Return LR scheduler or None. Override for custom scheduling."""
        return None

    def _forward(self, features: Tensor) -> Tensor:
        """Forward pass on features. Override for multi-input networks."""
        return self._net(features)

    @property
    def options(self) -> Options:
        """Current execution options."""
        return self._options

    def with_options(self, options: Options) -> Self:
        """Return copy with new options (immutable pattern)."""
        import copy

        new = copy.copy(self)
        new._options = options
        return new

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass with feature extraction."""
        features = self._extract_features(x)
        return self._forward(features)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step with automatic W&B logging."""
        if self._criterion is None:
            msg = "Training requires lr parameter in __init__"
            raise RuntimeError(msg)

        inputs, targets = batch
        features = self._extract_features(inputs)
        outputs = self._forward(features)
        loss = self._criterion(outputs, targets)

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
            "train/lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Validation step with automatic W&B logging."""
        if self._criterion is None:
            msg = "Validation requires lr parameter in __init__"
            raise RuntimeError(msg)

        inputs, targets = batch
        features = self._extract_features(inputs)
        outputs = self._forward(features)
        loss = self._criterion(outputs, targets)

        batch_size = targets.size(0)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Test step with automatic logging."""
        if self._criterion is None:
            msg = "Testing requires lr parameter in __init__"
            raise RuntimeError(msg)

        inputs, targets = batch
        features = self._extract_features(inputs)
        outputs = self._forward(features)
        loss = self._criterion(outputs, targets)

        batch_size = targets.size(0)
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and optional scheduler."""
        if self._lr is None:
            msg = "configure_optimizers requires lr parameter in __init__"
            raise RuntimeError(msg)

        optimizer = self._configure_optimizer(self._lr)

        if self._options.compile_optimizer and self._options.compile != CompileMode.OFF:
            optimizer = cast(Any, torch.compile)(optimizer)

        scheduler = self._configure_scheduler(optimizer)

        if scheduler is None:
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    @property
    def is_trained(self) -> bool:
        """Whether model has been trained."""
        return self.trainer is not None and self.trainer.state.finished

    def configure_model(self) -> None:
        """Configure model parallelization, compilation, and optimization.

        Called before training starts. Override to apply:
        - torch.compile for model optimization
        - Distributed tensor APIs (FSDP2, tensor parallelism)
        - TorchAO APIs (FP8 quantization)
        - Custom model transformations

        Access device mesh via self.device_mesh when using ModelParallelStrategy.

        Example:
            def configure_model(self) -> None:
                if self._options.compile != CompileMode.OFF:
                    self._net = torch.compile(
                        self._net,
                        mode=self._options.compile.value
                    )
        """

    def benchmark_latency(
        self,
        input_tensor: Tensor,
        n_runs: int = 100,
        warmup: int = 10,
    ) -> dict[str, float]:
        """Benchmark inference latency of the raw network.

        Runs _net directly, bypassing _extract_features and _forward.

        Args:
            input_tensor: Sample input (should be features, not raw input)
            n_runs: Number of timed runs
            warmup: Number of warmup runs (not timed)

        Returns:
            Dict with mean_ms, std_ms, min_ms, max_ms, median_ms

        Raises:
            RuntimeError: If options are invalid for benchmarking
        """
        result = self._options.validate_for_benchmark()
        if result.is_error():
            err = result.error
            msg = f"Invalid config for benchmarking: {err.field}={err.actual}"
            if err.remediation:
                msg += f". {err.remediation}"
            raise RuntimeError(msg)

        self.eval()
        input_tensor = input_tensor.to(self.device)

        with torch.inference_mode():
            for _ in range(warmup):
                self._net(input_tensor)

        times: list[float] = []
        with torch.inference_mode():
            for _ in range(n_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                self._net(input_tensor)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)

        times_t = torch.tensor(times)
        stats = {
            "mean_ms": float(times_t.mean()),
            "std_ms": float(times_t.std()),
            "min_ms": float(times_t.min()),
            "max_ms": float(times_t.max()),
            "median_ms": float(times_t.median()),
            "n_runs": n_runs,
        }

        if self.logger is not None:
            self.logger.log_metrics({f"latency/{k}": v for k, v in stats.items()})

        return stats

    def profile_memory(self, input_tensor: Tensor) -> dict[str, int]:
        """Profile memory usage of the raw network.

        Runs _net directly, bypassing _extract_features and _forward.

        Args:
            input_tensor: Sample input (should be features, not raw input)

        Returns:
            Dict with memory stats in bytes
        """
        self.eval()
        input_tensor = input_tensor.to(self.device)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            with torch.inference_mode():
                self._net(input_tensor)

            stats = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_bytes": torch.cuda.max_memory_allocated(),
            }
        else:
            param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
            buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
            stats = {
                "param_bytes": param_bytes,
                "buffer_bytes": buffer_bytes,
                "total_bytes": param_bytes + buffer_bytes,
            }

        if self.logger is not None:
            self.logger.log_metrics({f"memory/{k}": v for k, v in stats.items()})

        return stats

    @property
    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Return model summary string."""
        return (
            f"Model: {self.__class__.__name__}\n"
            f"Parameters: {self.param_count:,}\n"
            f"Device: {self.device}"
        )

    def save(self, path: str | Path) -> None:
        """Save model checkpoint.

        Args:
            path: Destination file path
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hparams": dict(self.hparams) if hasattr(self, "hparams") else {},
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            path: Source file path
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(checkpoint["state_dict"])
