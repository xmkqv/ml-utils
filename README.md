# ml-utils

Typed PyTorch Lightning utilities for training, evaluation, profiling, and
experiment tracking. Subclass one base module, supply a network, and inherit
typed metrics, execution options that carry their own rationale, latency and
memory profiling, and Weights & Biases tracking.

[![ci](https://github.com/xmkqv/ml-utils/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/xmkqv/ml-utils/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

The package is alpha. The API may change, and several execution presets assume
recent NVIDIA hardware.

## quickstart

Python 3.13 or newer, with [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/xmkqv/ml-utils.git
cd ml-utils
uv sync --locked --extra dev
uv run --extra dev pytest
```

Nothing here needs a Weights & Biases account. Pass `logger=False` to
`create_trainer` and every example below runs offline on CPU.

### train and evaluate a classifier

`MlClassifier` supplies the training, validation, and test steps. A subclass
supplies the network.

```python
import torch
from torch import nn

from ml_utils.data import TensorDataModule, stratified_split
from ml_utils.nets.classifier import MlClassifier
from ml_utils.runs.train import create_trainer


class LinearClassifier(MlClassifier):
    def __init__(self, input_size: int, class_names: tuple[str, ...]) -> None:
        self.input_size = input_size
        super().__init__(len(class_names), class_names, lr=0.05)

    def _build_net(self) -> nn.Module:
        return nn.Linear(self.input_size, self.num_classes)


torch.manual_seed(0)
centres = torch.randn(4, 16) * 0.6
labels = torch.randint(0, 4, (512,))
inputs = centres[labels] + torch.randn(512, 16)

train_x, train_y, val_x, val_y = stratified_split(inputs, labels, train_ratio=0.8)
data = TensorDataModule(train_x, train_y, val_x, val_y, batch_size=32)

model = LinearClassifier(16, ("a", "b", "c", "d"))
trainer = create_trainer(
    "example", max_epochs=20, logger=False, enable_progress_bar=False
)
trainer.fit(model, data)

result = model.evaluate(val_x, val_y, log=False)
print(f"accuracy {result.accuracy:.3f}  f1_macro {result.f1_macro:.3f}")
print(f"per class f1 {tuple(round(f, 3) for f in result.f1_per_class)}")
```

```
accuracy 0.856  f1_macro 0.852
per class f1 (0.836, 0.889, 0.739, 0.943)
```

`evaluate` returns a frozen `ClassificationResult`: accuracy, macro and
per-class F1, precision, recall, the confusion matrix, and the class names.
Individual metrics are reachable as `model.metrics.f1_per_class.calc(...)`.

### declare execution options and profile

`Options` records precision, compile mode, and CUDA graph choices together with
the reason for each. It converts to `Trainer` kwargs, to a W&B config, and back,
and it refuses to be used for a latency benchmark when compilation would distort
the measurement.

```python
import torch
from torch import nn

from ml_utils.nets.base import MlBaseModule
from ml_utils.runs.options import Options


class Mlp(MlBaseModule):
    def _build_net(self) -> nn.Module:
        return nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4))


model = Mlp(options=Options.for_benchmark())
sample = torch.randn(32, 16)

print(model.param_count)
latency = model.benchmark_latency(sample, n_runs=20, warmup=5)
print({k: round(v, 4) for k, v in latency.items()})
print(model.profile_memory(sample))

training = Options.for_training()
print(training.trainer_kwargs())
print(training.precision_rationale)

refused = training.validate_for_benchmark()
print(refused.is_error(), refused.error.remediation)
```

```
1348
{'mean_ms': 0.0187, 'std_ms': 0.0176, 'min_ms': 0.0129, 'max_ms': 0.093, 'median_ms': 0.0143, 'n_runs': 20}
{'param_bytes': 5392, 'buffer_bytes': 0, 'total_bytes': 5392}
{'precision': 'bf16-mixed', 'torch_compile': True, 'torch_compile_mode': 'reduce-overhead'}
OptionsRationale(choice='bf16', reason='2x throughput, stable on Ampere+ without grad scaler', citation='Axiom 2: Lower is Faster')
True Use Options.for_benchmark()
```

The latency figures are whatever the machine gives. Presets are `for_benchmark`, `for_training`, `for_production`, and
`for_distributed`. `create_trainer(options=...)` applies them.

### swap the loss and configure a sweep

`_configure_loss` is the hook for class imbalance. `create_sweep_config` and its
narrower variants emit dictionaries for `wandb.sweep`.

```python
import torch
from torch import Tensor, nn

from ml_utils.loss import FocalLoss
from ml_utils.nets.classifier import MlClassifier
from ml_utils.runs.sweep import create_sweep_config


class Imbalanced(MlClassifier):
    def _build_net(self) -> nn.Module:
        return nn.Linear(16, self.num_classes)

    def _configure_loss(self, class_weights: Tensor | None = None) -> nn.Module:
        return FocalLoss(gamma=2.0, alpha=class_weights)


weights = torch.tensor([0.6, 0.3, 0.1])
model = Imbalanced(3, ("rare", "common", "dominant"), class_weights=weights)

sweep = create_sweep_config(
    method="bayes",
    metric_name="val/f1",
    metric_goal="maximize",
    parameters={
        "lr": {"min": 1e-4, "max": 1e-1},
        "batch_size": {"values": [16, 32, 64]},
    },
)
print(sweep)
```

```
{'method': 'bayes', 'metric': {'name': 'val/f1', 'goal': 'maximize'}, 'parameters': {'lr': {'min': 0.0001, 'max': 0.1}, 'batch_size': {'values': [16, 32, 64]}}, 'early_terminate': {'type': 'hyperband', 'min_iter': 3, 'eta': 2}}
```

## modules

- `ml_utils.data` provides NumPy and PyTorch stratified splits, a tensor-backed
  Lightning data module, and a TOML-configured dataset registry
- `ml_utils.nets` contains the Lightning base module and the classifier, with
  profiling, serialization, and misclassification analysis
- `ml_utils.runs` contains execution options, trainer construction, and sweep
  configuration
- `ml_utils.utils` contains typed classification metrics and Weights & Biases
  artifact helpers
- `ml_utils.loss` contains focal loss and scaled dot-product and multi-query
  attention layers

## limits

- alpha library rather than a stable public API
- targets Python 3.13 or newer and current PyTorch and Lightning releases
- the production and training presets assume recent NVIDIA hardware and should
  be checked against the available accelerator before use
- the Weights & Biases helpers need an authenticated account for online logging
- no experiment logs, checkpoints, or trained weights ship with the repository

## development

The same commands CI runs.

```sh
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev pyright
uv build
```

Tests use small synthetic datasets and run on CPU.

## license

[MIT](LICENSE).
