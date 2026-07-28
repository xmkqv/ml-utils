# Comprehensive run options for nets

from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

import torch
from expression import Error, Ok, Result


class Precision(Enum):
    """Numeric precision for model execution."""

    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"


class CompileMode(Enum):
    """torch.compile optimization modes."""

    OFF = "off"
    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"


@dataclass(frozen=True, slots=True)
class OptionsRationale:
    """Structured reasoning for an execution option choice."""

    choice: str
    reason: str
    citation: str | None = None


@dataclass(frozen=True, slots=True)
class OptionsError:
    """Execution options validation error."""

    field: str
    expected: str
    actual: str
    remediation: str | None = None


@dataclass(frozen=True, slots=True)
class CompileConfig:
    """Advanced torch.compile configuration."""

    backend: str = "inductor"
    fullgraph: bool = False
    dynamic: bool | None = None


@dataclass(frozen=True, slots=True)
class Options:
    """PyTorch execution options with documented reasoning.

    Attached to models to declare execution choices and their rationale.
    Validated at benchmark and optimization checkpoints.
    """

    compile: CompileMode = CompileMode.OFF
    precision: Precision = Precision.FP32
    cuda_graphs: bool = False
    compile_optimizer: bool = False
    compile_config: CompileConfig | None = None

    compile_rationale: OptionsRationale | None = None
    precision_rationale: OptionsRationale | None = None
    cuda_graphs_rationale: OptionsRationale | None = None
    compile_optimizer_rationale: OptionsRationale | None = None

    @classmethod
    def for_benchmark(cls) -> Self:
        """Configuration for raw latency benchmarking."""
        return cls(
            compile=CompileMode.OFF,
            precision=Precision.FP32,
            cuda_graphs=False,
            compile_rationale=OptionsRationale(
                choice="off",
                reason="Measure raw kernel latency without JIT overhead",
                citation="Profile before optimizing",
            ),
            precision_rationale=OptionsRationale(
                choice="fp32",
                reason="Reference precision for numerical validation",
            ),
            cuda_graphs_rationale=OptionsRationale(
                choice="disabled",
                reason="Measure kernel launch overhead realistically",
            ),
        )

    @classmethod
    def for_production(cls) -> Self:
        """Configuration for production inference."""
        return cls(
            compile=CompileMode.MAX_AUTOTUNE,
            precision=Precision.BF16,
            cuda_graphs=True,
            compile_rationale=OptionsRationale(
                choice="max-autotune",
                reason="Maximum inference throughput, amortize compile over lifetime",
                citation="Axiom 1: JIT Everything",
            ),
            precision_rationale=OptionsRationale(
                choice="bf16",
                reason="16x throughput vs FP32, no gradient scaling needed on Ampere+",
                citation="Axiom 2: Lower is Faster",
            ),
            cuda_graphs_rationale=OptionsRationale(
                choice="enabled",
                reason="Eliminate kernel launch overhead for static shapes",
                citation="Axiom 5: Capture and Replay",
            ),
        )

    @classmethod
    def for_training(cls) -> Self:
        """Configuration for training."""
        return cls(
            compile=CompileMode.REDUCE_OVERHEAD,
            precision=Precision.BF16,
            cuda_graphs=False,
            compile_optimizer=True,
            compile_rationale=OptionsRationale(
                choice="reduce-overhead",
                reason="Balanced compile/run tradeoff for limited training window",
                citation="Axiom 1: JIT Everything",
            ),
            precision_rationale=OptionsRationale(
                choice="bf16",
                reason="2x throughput, stable on Ampere+ without grad scaler",
                citation="Axiom 2: Lower is Faster",
            ),
            cuda_graphs_rationale=OptionsRationale(
                choice="disabled",
                reason="Training has dynamic control flow incompatible with graphs",
            ),
            compile_optimizer_rationale=OptionsRationale(
                choice="enabled",
                reason="Additional 10-20% training speedup via optimizer compilation",
                citation="PyTorch 2.2+ optimizer compilation",
            ),
        )

    @classmethod
    def for_distributed(cls, strategy: str = "fsdp") -> Self:
        """Configuration for distributed training.

        Args:
            strategy: Distributed strategy ("ddp", "fsdp", "fsdp2")
        """
        return cls(
            compile=CompileMode.REDUCE_OVERHEAD,
            precision=Precision.BF16,
            cuda_graphs=False,
            compile_optimizer=True,
            compile_rationale=OptionsRationale(
                choice="reduce-overhead",
                reason=f"Balanced compile time for {strategy} distributed training",
                citation="Axiom 1: JIT Everything",
            ),
            precision_rationale=OptionsRationale(
                choice="bf16",
                reason=(
                    "Mixed precision reduces communication overhead "
                    "in distributed training"
                ),
                citation="Axiom 2: Lower is Faster",
            ),
            cuda_graphs_rationale=OptionsRationale(
                choice="disabled",
                reason="CUDA graphs incompatible with distributed collectives",
            ),
            compile_optimizer_rationale=OptionsRationale(
                choice="enabled",
                reason="Optimizer compilation works with FSDP2",
            ),
        )

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        """Return dtype for torch.autocast, or None if FP32 (no autocast)."""
        match self.precision:
            case Precision.BF16:
                return torch.bfloat16
            case Precision.FP16:
                return torch.float16
            case Precision.FP32:
                return None

    def validate_for_benchmark(self) -> Result[None, OptionsError]:
        """Validate config is suitable for raw latency benchmarking."""
        if self.compile != CompileMode.OFF:
            return Error(
                OptionsError(
                    field="compile",
                    expected="off",
                    actual=self.compile.value,
                    remediation="Use Options.for_benchmark()",
                )
            )
        if self.cuda_graphs:
            return Error(
                OptionsError(
                    field="cuda_graphs",
                    expected="False",
                    actual="True",
                    remediation="Use Options.for_benchmark()",
                )
            )
        return Ok(None)

    def trainer_kwargs(self) -> dict[str, Any]:
        """Generate PyTorch Lightning Trainer kwargs from options.

        Returns:
            Dict of Trainer constructor kwargs for precision and compilation.
        """
        kwargs: dict[str, Any] = {}

        match self.precision:
            case Precision.BF16:
                kwargs["precision"] = "bf16-mixed"
            case Precision.FP16:
                kwargs["precision"] = "16-mixed"
            case Precision.FP32:
                kwargs["precision"] = "32-true"

        if self.compile != CompileMode.OFF:
            kwargs["torch_compile"] = True
            if self.compile_config:
                kwargs["torch_compile_kwargs"] = {
                    "mode": self.compile.value,
                    "backend": self.compile_config.backend,
                    "fullgraph": self.compile_config.fullgraph,
                    "dynamic": self.compile_config.dynamic,
                }
            else:
                kwargs["torch_compile_mode"] = self.compile.value

        return kwargs

    def to_wandb_config(self) -> dict[str, Any]:
        """Export options as W&B config dictionary.

        Returns:
            Config dict suitable for wandb.init(config=...) or WandbLogger
        """
        config: dict[str, Any] = {
            "compile_mode": self.compile.value,
            "precision": self.precision.value,
            "cuda_graphs": self.cuda_graphs,
            "compile_optimizer": self.compile_optimizer,
        }

        if self.compile_config:
            config["compile_backend"] = self.compile_config.backend
            config["compile_fullgraph"] = self.compile_config.fullgraph
            config["compile_dynamic"] = self.compile_config.dynamic

        if self.compile_rationale:
            config["compile_rationale"] = {
                "choice": self.compile_rationale.choice,
                "reason": self.compile_rationale.reason,
            }
            if self.compile_rationale.citation:
                config["compile_citation"] = self.compile_rationale.citation

        if self.precision_rationale:
            config["precision_rationale"] = {
                "choice": self.precision_rationale.choice,
                "reason": self.precision_rationale.reason,
            }
            if self.precision_rationale.citation:
                config["precision_citation"] = self.precision_rationale.citation

        return config

    @classmethod
    def from_wandb_config(cls, config: dict[str, Any]) -> Self:
        """Reconstruct Options from W&B config.

        Useful for reproducing experiments from logged runs.
        """
        compile_config = None
        if "compile_backend" in config:
            compile_config = CompileConfig(
                backend=config.get("compile_backend", "inductor"),
                fullgraph=config.get("compile_fullgraph", False),
                dynamic=config.get("compile_dynamic"),
            )

        return cls(
            compile=CompileMode(config.get("compile_mode", "off")),
            precision=Precision(config.get("precision", "fp32")),
            cuda_graphs=config.get("cuda_graphs", False),
            compile_optimizer=config.get("compile_optimizer", False),
            compile_config=compile_config,
        )
