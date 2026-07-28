"""W&B tracking utilities."""

from pathlib import Path
from typing import Any

import wandb
from pytorch_lightning.loggers import WandbLogger


def create_wandb_logger(
    project: str,
    name: str | None = None,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
    save_dir: str | Path = ".",
    offline: bool = False,
    **kwargs: Any,
) -> WandbLogger:
    """Create a configured W&B logger."""
    return WandbLogger(
        project=project,
        name=name,
        tags=tags,
        config=config,
        save_dir=str(save_dir),
        offline=offline,
        log_model=True,
        **kwargs,
    )


def log_model_artifact(
    model_path: str | Path,
    name: str,
    metadata: dict[str, Any] | None = None,
    aliases: list[str] | None = None,
    dataset_artifact: str | None = None,
) -> wandb.Artifact | None:
    """Log model checkpoint as W&B artifact with enhanced metadata.

    Args:
        model_path: Path to saved model
        name: Artifact name
        metadata: Additional metadata (metrics, hyperparameters, etc.)
        aliases: Artifact aliases like ["latest", "best", "production"]
        dataset_artifact: Reference to dataset artifact used for training

    Returns:
        Logged artifact instance, or None if no active run

    Example:
        >>> log_model_artifact(
        ...     model_path="checkpoints/best.pt",
        ...     name="classifier-v1",
        ...     metadata={"val_f1": 0.95, "val_acc": 0.94},
        ...     aliases=["latest", "best"],
        ...     dataset_artifact="my-dataset:v3"
        ... )
    """
    if wandb.run is None:
        return None

    from pathlib import Path as PathLib

    model_path = PathLib(model_path)

    # Enhance metadata with file info
    enhanced_metadata = {
        "file_size_mb": model_path.stat().st_size / (1024 * 1024),
        "file_name": model_path.name,
        **(metadata or {}),
    }

    artifact = wandb.Artifact(
        name=name,
        type="model",
        metadata=enhanced_metadata,
    )
    artifact.add_file(str(model_path))

    # Link to dataset artifact if provided
    if dataset_artifact:
        try:
            dataset = wandb.use_artifact(dataset_artifact)
            artifact.metadata["dataset_artifact"] = dataset_artifact
            artifact.metadata["dataset_digest"] = dataset.digest
        except Exception:
            # Dataset artifact not found, continue without linking
            pass

    # Log with aliases
    wandb.log_artifact(artifact, aliases=aliases or [])

    return artifact


def load_model_artifact(
    artifact_name: str,
    alias: str = "latest",
    download_dir: str | Path | None = None,
) -> Path:
    """Download and return path to model artifact.

    Args:
        artifact_name: Artifact name (e.g., "my-model")
        alias: Artifact alias (e.g., "latest", "best", "production")
        download_dir: Directory to download to (default: ./artifacts)

    Returns:
        Path to downloaded model file

    Raises:
        RuntimeError: If no active W&B run
        FileNotFoundError: If no model file found in artifact

    Example:
        >>> model_path = load_model_artifact("classifier-v1", alias="best")
        >>> model.load(model_path)
    """
    from pathlib import Path as PathLib

    if wandb.run is None:
        msg = "No active W&B run. Call wandb.init() first."
        raise RuntimeError(msg)

    artifact_ref = f"{artifact_name}:{alias}"
    artifact = wandb.use_artifact(artifact_ref)

    download_path = str(download_dir) if download_dir else "./artifacts"
    artifact_dir = artifact.download(root=download_path)

    # Find the model file
    artifact_path = PathLib(artifact_dir)
    model_files = list(artifact_path.glob("*.pt")) + list(artifact_path.glob("*.pth"))

    if not model_files:
        msg = f"No model file found in artifact {artifact_ref}"
        raise FileNotFoundError(msg)

    return model_files[0]


def log_hyperparameters(hparams: dict[str, Any]) -> None:
    """Log hyperparameters to W&B config.

    Args:
        hparams: Hyperparameters dictionary
    """
    if wandb.run is not None:
        wandb.config.update(hparams)


def log_dataset_artifact(
    name: str,
    train_samples: int,
    val_samples: int,
    num_features: int,
    num_classes: int,
    metadata: dict[str, Any] | None = None,
    files: list[str | Path] | None = None,
) -> wandb.Artifact | None:
    """Log dataset as W&B artifact.

    Args:
        name: Artifact name
        train_samples: Number of training samples
        val_samples: Number of validation samples
        num_features: Number of input features
        num_classes: Number of output classes
        metadata: Additional metadata
        files: Optional list of files to include

    Returns:
        Logged artifact instance, or None if no active run
    """
    if wandb.run is None:
        return None

    enhanced_metadata = {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "num_features": num_features,
        "num_classes": num_classes,
        **(metadata or {}),
    }

    artifact = wandb.Artifact(
        name=name,
        type="dataset",
        metadata=enhanced_metadata,
    )

    if files:
        for file_path in files:
            artifact.add_file(str(file_path))

    wandb.log_artifact(artifact)
    return artifact
