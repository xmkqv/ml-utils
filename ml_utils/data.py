# Extended data loaders

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import pytorch_lightning as pl
import torch
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

"""Data loading utilities for Lightning.

Provides:
- TensorDataModule: DataModule for tensor data
- stratified_split: Stratified train/test split for tensors
"""


class TensorDataModule(pl.LightningDataModule):
    """DataModule for tensor data.

    Pure tensor interface - no numpy.

    Args:
        train_inputs: Training inputs [N, ...]
        train_targets: Training targets [N, ...]
            (float for regression, long for classification)
        val_inputs: Optional validation inputs
        val_targets: Optional validation targets
        batch_size: Mini-batch size
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer
    """

    def __init__(
        self,
        train_inputs: Tensor,
        train_targets: Tensor,
        val_inputs: Tensor | None = None,
        val_targets: Tensor | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self._train_inputs = train_inputs.float()
        self._train_targets = train_targets
        self._val_inputs = val_inputs.float() if val_inputs is not None else None
        self._val_targets = val_targets
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory

    def setup(self, stage: str | None = None) -> None:
        """Setup data - no-op since data is already prepared."""

    def train_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """Create training DataLoader."""
        dataset = TensorDataset(self._train_inputs, self._train_targets)
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, ...]] | None:
        """Create validation DataLoader if validation data exists."""
        if self._val_inputs is None or self._val_targets is None:
            return None
        dataset = TensorDataset(self._val_inputs, self._val_targets)
        return DataLoader(
            dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
        )


def stratified_split(
    data: Tensor,
    labels: Tensor,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Stratified train/test split for tensors.

    Preserves class distribution in both splits.

    Args:
        data: Input data [N, ...]
        labels: Class labels [N] as long tensor
        train_ratio: Fraction for training set
        seed: Random seed for reproducibility

    Returns:
        (train_data, train_labels, test_data, test_labels)
    """
    generator = torch.Generator().manual_seed(seed)
    num_classes = int(labels.max().item()) + 1

    train_indices: list[int] = []
    test_indices: list[int] = []

    for c in range(num_classes):
        class_indices = (labels == c).nonzero(as_tuple=True)[0]
        n_class = class_indices.size(0)
        n_train = int(n_class * train_ratio)

        perm = torch.randperm(n_class, generator=generator)
        train_indices.extend(class_indices[perm[:n_train]].tolist())
        test_indices.extend(class_indices[perm[n_train:]].tolist())

    train_idx = torch.tensor(train_indices)
    test_idx = torch.tensor(test_indices)

    return data[train_idx], labels[train_idx], data[test_idx], labels[test_idx]


class InvariantError(Exception):
    """Raised when dataset invariants are violated."""


class RegistryError(Exception):
    """Raised when registry operations fail."""


@dataclass(frozen=True, slots=True)
class Info:
    """Immutable dataset metadata."""

    name: str
    description: str
    source_url: str
    license: str
    original_classes: int
    canonical_classes: int
    sample_count: int
    known_issues: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Loaded:
    """Loaded dataset with enforced invariants.

    Invariants:
    - data.dtype == np.float64
    - labels.dtype == np.int64
    - len(data) == len(labels)
    """

    data: NDArray[np.float64]
    labels: NDArray[np.int64]
    info: Info
    stats: dict[str, int]

    def __post_init__(self) -> None:
        """Validate invariants."""
        if self.data.dtype != np.float64:
            msg = f"data.dtype must be float64, got {self.data.dtype}"
            raise InvariantError(msg)
        if self.labels.dtype != np.int64:
            msg = f"labels.dtype must be int64, got {self.labels.dtype}"
            raise InvariantError(msg)
        if len(self.data) != len(self.labels):
            msg = f"data/labels length mismatch: {len(self.data)} vs {len(self.labels)}"
            raise InvariantError(msg)

    def to_tensors(self) -> tuple[Tensor, Tensor]:
        """Convert to pure tensor format.

        Returns:
            (data_tensor, labels_tensor) as float32 and long tensors
        """
        import torch

        return (
            torch.from_numpy(self.data).float(),
            torch.from_numpy(self.labels).long(),
        )


@runtime_checkable
class Loader(Protocol):
    """Protocol for dataset loaders."""

    def info(self) -> Info:
        """Return dataset metadata without loading data."""
        ...

    def load(self, resample_length: int | None = None) -> Loaded:
        """Load full dataset."""
        ...

    def load_split(
        self,
        test_fraction: float = 0.2,
        seed: int = 42,
        resample_length: int | None = None,
    ) -> tuple[Loaded, Loaded]:
        """Load train/test split."""
        ...


class LoaderFactory(Protocol):
    def __call__(self, config: dict[str, Any]) -> Loader: ...


def stratified_indices(
    labels: NDArray[np.int64],
    test_fraction: float,
    seed: int = 42,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Compute stratified train/test split indices.

    Args:
        labels: Integer class labels
        test_fraction: Fraction of data for test set
        seed: Random seed for reproducibility

    Returns:
        (train_indices, test_indices) as numpy arrays
    """
    indices = np.arange(len(labels))
    train_idx, test_idx = cast(
        tuple[NDArray[np.intp], NDArray[np.intp]],
        train_test_split(
            indices,
            test_size=test_fraction,
            stratify=labels,
            random_state=seed,
        ),
    )
    return train_idx.astype(np.intp), test_idx.astype(np.intp)


class Registry:
    """Registry for loading datasets from TOML configuration.

    Config format:
        [datasets.name]
        loader = "module.path:ClassName"
        path = "/path/to/data"
        description = "Dataset description"
        # ... additional loader-specific config
    """

    def __init__(self, config_path: str | Path) -> None:
        """Load registry from TOML config file."""
        self._config_path = Path(config_path)
        if not self._config_path.exists():
            msg = f"Config file not found: {config_path}"
            raise RegistryError(msg)

        with self._config_path.open("rb") as f:
            self._config = tomllib.load(f)

        self._datasets: dict[str, dict[str, Any]] = self._config.get("datasets", {})
        self._loaders: dict[str, Loader] = {}

    def list_datasets(self) -> list[str]:
        """List available dataset names."""
        return list(self._datasets.keys())

    def get_info(self, name: str) -> Info:
        """Get dataset metadata without loading."""
        return self.get_loader(name).info()

    def get_loader(self, name: str) -> Loader:
        """Get or create loader for named dataset."""
        if name in self._loaders:
            return self._loaders[name]

        if name not in self._datasets:
            available = ", ".join(self.list_datasets())
            msg = f"Unknown dataset '{name}'. Available: {available}"
            raise RegistryError(msg)

        config = dict(self._datasets[name])
        loader_ref = config.pop("loader", None)
        if not loader_ref:
            msg = f"Dataset '{name}' missing 'loader' field"
            raise RegistryError(msg)

        loader_cls = self._import_loader(loader_ref)
        loader = loader_cls(config)
        self._loaders[name] = loader
        return loader

    def _import_loader(self, ref: str) -> LoaderFactory:
        """Import loader class from qualified reference."""
        if ":" in ref:
            module_path, class_name = ref.rsplit(":", 1)
        else:
            module_path, class_name = ref.rsplit(".", 1)

        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            msg = f"Failed to import loader '{ref}': {e}"
            raise RegistryError(msg) from e

        return cast(LoaderFactory, cls)
