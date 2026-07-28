# ml-utils

`ml-utils` is a small Python package for building and evaluating PyTorch
Lightning models. It collects the training code that tends to repeat across
experiments: tensor data modules, stratified splits, execution options, model
profiling, classification metrics, focal loss, and Weights & Biases helpers.

The package is experimental. Its API may change, and several execution presets
assume recent NVIDIA hardware.

## What is included

- `ml_utils.data` provides NumPy and PyTorch stratified splits, a tensor-backed
  Lightning data module, and a configurable dataset registry
- `ml_utils.nets` contains reusable Lightning base classes for regression and
  classification models
- `ml_utils.runs` contains explicit precision and compilation options, trainer
  construction, and sweep configuration
- `ml_utils.utils` contains typed classification metrics and optional Weights &
  Biases artifact helpers
- `ml_utils.loss` contains focal loss and attention layers

## Setup

The project requires Python 3.13 and uses [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/xmkqv/ml-utils.git
cd ml-utils
uv sync --locked --extra dev
```

## Example

Subclass `MlClassifier` to supply the network used for a classification task:

```python
import torch
from torch import Tensor, nn

from ml_utils.data import TensorDataModule
from ml_utils.nets.classifier import MlClassifier
from ml_utils.runs.train import create_trainer


class LinearClassifier(MlClassifier):
    def __init__(self, input_size: int, class_names: tuple[str, ...]) -> None:
        self.input_size = input_size
        super().__init__(len(class_names), class_names)

    def _build_net(self) -> nn.Module:
        return nn.Linear(self.input_size, self.num_classes)


inputs = torch.randn(128, 16)
labels = torch.randint(0, 4, (128,))
data = TensorDataModule(inputs, labels, batch_size=32)
model = LinearClassifier(16, ("a", "b", "c", "d"))
trainer = create_trainer("example", max_epochs=5, logger=False)
trainer.fit(model, data)
```

`create_trainer` enables Weights & Biases when `logger=True`. Set `logger=False`
for a local run with no external tracking.

## Development

```sh
uv run --extra dev pytest
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev pyright
uv build
```

Tests use small synthetic datasets and run on CPU. The repository does not
include experiment logs, model checkpoints, or trained weights.

## Limits

- The package targets Python 3.13 and current PyTorch and Lightning releases
- Production and training presets need to be checked against the available
  accelerator before use
- Weights & Biases helpers require an authenticated account for online logging
- This is an alpha library rather than a stable public API

## Reuse

Released under the [MIT License](LICENSE).
