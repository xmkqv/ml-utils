# Singular sweep creator

from typing import Any


def create_sweep_config(
    method: str = "bayes",
    metric_name: str = "val/loss",
    metric_goal: str = "minimize",
    parameters: dict[str, Any] | None = None,
    early_terminate: bool = True,
) -> dict[str, Any]:
    """Create W&B sweep configuration.

    Args:
        method: Search method - 'bayes', 'random', or 'grid'
        metric_name: Metric to optimize
        metric_goal: 'minimize' or 'maximize'
        parameters: Hyperparameter search space
        early_terminate: Enable early termination (Hyperband)

    Returns:
        Sweep configuration dict for wandb.sweep()

    Example:
        >>> sweep_config = create_sweep_config(
        ...     method="bayes",
        ...     metric_name="val/f1",
        ...     metric_goal="maximize",
        ...     parameters={
        ...         "lr": {"min": 0.0001, "max": 0.1},
        ...         "batch_size": {"values": [16, 32, 64, 128]},
        ...     }
        ... )
        >>> sweep_id = wandb.sweep(sweep_config, project="my-project")
        >>> wandb.agent(sweep_id, function=train_fn, count=20)
    """
    config: dict[str, Any] = {
        "method": method,
        "metric": {
            "name": metric_name,
            "goal": metric_goal,
        },
        "parameters": parameters or {},
    }

    if early_terminate and method == "bayes":
        config["early_terminate"] = {
            "type": "hyperband",
            "min_iter": 3,
            "eta": 2,
        }

    return config


def create_lr_sweep(
    min_lr: float = 1e-5,
    max_lr: float = 1e-1,
    num_trials: int = 20,
) -> dict[str, Any]:
    """Create learning rate sweep configuration.

    Useful for finding optimal learning rate before full training.

    Args:
        min_lr: Minimum learning rate to search
        max_lr: Maximum learning rate to search
        num_trials: Number of trials (unused, for documentation)

    Returns:
        Sweep configuration dict
    """
    return create_sweep_config(
        method="random",
        metric_name="val/loss",
        metric_goal="minimize",
        parameters={
            "lr": {
                "distribution": "log_uniform_values",
                "min": min_lr,
                "max": max_lr,
            }
        },
        early_terminate=True,
    )


def create_architecture_sweep(
    hidden_sizes: list[list[int]],
    dropout_rates: list[float],
    lr_min: float = 1e-4,
    lr_max: float = 1e-2,
) -> dict[str, Any]:
    """Create architecture hyperparameter sweep.

    Args:
        hidden_sizes: List of hidden layer configurations to try
        dropout_rates: List of dropout rates to try
        lr_min: Minimum learning rate
        lr_max: Maximum learning rate

    Returns:
        Sweep configuration dict
    """
    return create_sweep_config(
        method="bayes",
        metric_name="val/f1",
        metric_goal="maximize",
        parameters={
            "hidden_sizes": {"values": hidden_sizes},
            "dropout": {"values": dropout_rates},
            "lr": {
                "distribution": "log_uniform_values",
                "min": lr_min,
                "max": lr_max,
            },
        },
        early_terminate=True,
    )


def create_regularization_sweep(
    weight_decay_values: list[float] | None = None,
    dropout_values: list[float] | None = None,
) -> dict[str, Any]:
    """Create regularization hyperparameter sweep.

    Args:
        weight_decay_values: List of weight decay values to try
        dropout_values: List of dropout values to try

    Returns:
        Sweep configuration dict
    """
    parameters: dict[str, Any] = {}

    if weight_decay_values:
        parameters["weight_decay"] = {"values": weight_decay_values}
    else:
        parameters["weight_decay"] = {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-2,
        }

    if dropout_values:
        parameters["dropout"] = {"values": dropout_values}
    else:
        parameters["dropout"] = {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.5,
        }

    return create_sweep_config(
        method="bayes",
        metric_name="val/loss",
        metric_goal="minimize",
        parameters=parameters,
        early_terminate=True,
    )
