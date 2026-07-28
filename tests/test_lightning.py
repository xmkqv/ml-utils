"""Tests for ml_utils.lightning modules."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from ml_utils.runs.options import Options, Precision
from ml_utils.runs.train import (
    BaseModule,
    ClassificationResult,
    Classifier,
    DataModule,
    create_trainer,
    stratified_split,
)


class SimpleNet(nn.Module):
    """Minimal network for testing."""

    def __init__(self, in_features: int = 10, out_features: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class ConcreteBaseModule(BaseModule):
    """Concrete BaseModule for testing."""

    def __init__(
        self,
        in_features: int = 10,
        out_features: int = 2,
        seed: int = 42,
        lr: float = 0.01,
    ) -> None:
        self._in_features = in_features
        self._out_features = out_features
        super().__init__(seed=seed, lr=lr)

    def _build_net(self) -> nn.Module:
        return SimpleNet(self._in_features, self._out_features)

    def _extract_features(self, inputs: Tensor) -> Tensor:
        return inputs


class ConcreteClassifier(Classifier):
    """Concrete Classifier for testing."""

    def __init__(
        self,
        num_classes: int = 2,
        class_names: tuple[str, ...] = ("A", "B"),
        in_features: int = 10,
        seed: int = 42,
        lr: float = 0.01,
        class_weights: Tensor | None = None,
    ) -> None:
        self._in_features = in_features
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            seed=seed,
            lr=lr,
            class_weights=class_weights,
        )

    def _build_net(self) -> nn.Module:
        return SimpleNet(self._in_features, self.num_classes)

    def _extract_features(self, inputs: Tensor) -> Tensor:
        return inputs


class TestTrainableModule:
    """Tests for BaseModule."""

    def test_init_builds_net(self) -> None:
        model = ConcreteBaseModule()
        assert isinstance(model._net, SimpleNet)

    def test_forward_works(self) -> None:
        model = ConcreteBaseModule()
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 2)

    def test_param_count(self) -> None:
        model = ConcreteBaseModule()
        assert model.param_count == 10 * 2 + 2  # weights + bias

    def test_summary(self) -> None:
        model = ConcreteBaseModule()
        summary = model.summary()
        assert "ConcreteBaseModule" in summary
        assert "Parameters" in summary

    def test_with_options_returns_copy(self) -> None:
        model = ConcreteBaseModule()
        new_options = Options.for_training()
        new_model = model.with_options(new_options)
        assert new_model is not model
        assert new_model.options is new_options

    def test_benchmark_latency(self) -> None:
        model = ConcreteBaseModule()
        x = torch.randn(1, 10)
        stats = model.benchmark_latency(x, n_runs=5, warmup=1)
        assert "mean_ms" in stats
        assert "std_ms" in stats
        assert stats["n_runs"] == 5

    def test_benchmark_latency_rejects_invalid_options(self) -> None:
        model = ConcreteBaseModule()
        model = model.with_options(Options.for_production())
        x = torch.randn(1, 10)
        with pytest.raises(RuntimeError, match="Invalid config"):
            model.benchmark_latency(x)

    def test_profile_memory_cpu(self) -> None:
        model = ConcreteBaseModule()
        x = torch.randn(1, 10)
        stats = model.profile_memory(x)
        assert "param_bytes" in stats or "allocated_bytes" in stats

    def test_save_and_load(self) -> None:
        model = ConcreteBaseModule(seed=123)
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            model.save(path)

            model2 = ConcreteBaseModule(seed=456)
            model2.load(path)

            for k, v in model2.state_dict().items():
                assert torch.allclose(v, original_state[k])


class TestBaseModule:
    """Tests for BaseModule."""

    def test_forward_extracts_features(self) -> None:
        model = ConcreteBaseModule()
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 2)

    def test_training_step(self) -> None:
        model = ConcreteBaseModule()
        # Mock trainer with optimizer
        mock_trainer = MagicMock()
        mock_trainer.optimizers = [torch.optim.Adam(model.parameters())]
        model.trainer = mock_trainer
        model.log = MagicMock()  # Mock log to avoid trainer dependency

        batch = (torch.randn(4, 10), torch.randn(4, 2))
        loss = model.training_step(batch, 0)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_validation_step(self) -> None:
        model = ConcreteBaseModule()
        batch = (torch.randn(4, 10), torch.randn(4, 2))
        loss = model.validation_step(batch, 0)
        assert loss.shape == ()

    def test_configure_optimizers_default(self) -> None:
        model = ConcreteBaseModule()
        config = model.configure_optimizers()
        assert "optimizer" in config
        assert isinstance(config["optimizer"], torch.optim.Adam)

    def test_configure_loss_default_mse(self) -> None:
        model = ConcreteBaseModule()
        assert isinstance(model._criterion, nn.MSELoss)

    def test_custom_optimizer(self) -> None:
        class SGDModel(ConcreteBaseModule):
            def _configure_optimizer(self, lr: float) -> torch.optim.Optimizer:
                return torch.optim.SGD(self.parameters(), lr=lr)

        model = SGDModel()
        config = model.configure_optimizers()
        assert isinstance(config["optimizer"], torch.optim.SGD)

    def test_custom_scheduler(self) -> None:
        class ScheduledModel(ConcreteBaseModule):
            def _configure_scheduler(
                self, optimizer: torch.optim.Optimizer
            ) -> torch.optim.lr_scheduler.LRScheduler:
                return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        model = ScheduledModel()
        config = model.configure_optimizers()
        assert "lr_scheduler" in config
        assert isinstance(
            config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.StepLR
        )


class TestClassifier:
    """Tests for Classifier."""

    def test_init_validates_class_names(self) -> None:
        with pytest.raises(ValueError, match="class_names length"):
            ConcreteClassifier(num_classes=3, class_names=("A", "B"))

    def test_num_classes_property(self) -> None:
        model = ConcreteClassifier(num_classes=3, class_names=("A", "B", "C"))
        assert model.num_classes == 3

    def test_class_names_property(self) -> None:
        model = ConcreteClassifier(num_classes=2, class_names=("X", "Y"))
        assert model.class_names == ("X", "Y")

    def test_configure_loss_cross_entropy(self) -> None:
        model = ConcreteClassifier()
        assert isinstance(model._criterion, nn.CrossEntropyLoss)

    def test_configure_loss_with_weights(self) -> None:
        weights = torch.tensor([1.0, 2.0])
        model = ConcreteClassifier(class_weights=weights)
        assert model._criterion.weight is not None

    def test_training_step(self) -> None:
        model = ConcreteClassifier()
        # Mock trainer with optimizer
        mock_trainer = MagicMock()
        mock_trainer.optimizers = [torch.optim.Adam(model.parameters())]
        model.trainer = mock_trainer
        model.log = MagicMock()  # Mock log to avoid trainer dependency

        batch = (torch.randn(4, 10), torch.randint(0, 2, (4,)))
        loss = model.training_step(batch, 0)
        assert loss.shape == ()

    def test_validation_step(self) -> None:
        model = ConcreteClassifier()
        batch = (torch.randn(4, 10), torch.randint(0, 2, (4,)))
        loss = model.validation_step(batch, 0)
        assert loss.shape == ()

    def test_predict(self) -> None:
        model = ConcreteClassifier()
        x = torch.randn(8, 10)
        preds = model.predict(x)
        assert preds.shape == (8,)
        assert preds.dtype == torch.int64

    def test_predict_step(self) -> None:
        model = ConcreteClassifier()
        x = torch.randn(4, 10)
        preds = model.predict_step(x, 0)
        assert preds.shape == (4,)

    def test_evaluate(self) -> None:
        model = ConcreteClassifier()
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        result = model.evaluate(x, y, log=False)
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.accuracy <= 1.0
        assert result.confusion.shape == (2, 2)


class TestMetrics:
    """Tests for metrics system."""

    def test_accuracy_calc(self) -> None:
        model = ConcreteClassifier()
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])
        acc = model.metrics.accuracy.calc(preds, targets, log=False)
        assert acc == 1.0

    def test_f1_calc(self) -> None:
        model = ConcreteClassifier()
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])
        f1 = model.metrics.f1.calc(preds, targets, log=False)
        assert f1 == 1.0

    def test_metrics_calc_all(self) -> None:
        model = ConcreteClassifier()
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 1])
        result = model.metrics.calc(preds, targets, log=False)
        assert result.accuracy == 1.0
        assert result.f1_macro == 1.0
        assert result.confusion.shape == (2, 2)


class TestDataModule:
    """Tests for DataModule."""

    def test_train_dataloader(self) -> None:
        train_x = torch.randn(100, 10)
        train_y = torch.randint(0, 2, (100,))
        dm = DataModule(train_x, train_y, batch_size=32)
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert len(batch) == 2
        assert batch[0].shape[0] <= 32

    def test_val_dataloader_none_when_no_val_data(self) -> None:
        train_x = torch.randn(100, 10)
        train_y = torch.randint(0, 2, (100,))
        dm = DataModule(train_x, train_y)
        dm.setup()

        assert dm.val_dataloader() is None

    def test_val_dataloader_with_val_data(self) -> None:
        train_x = torch.randn(100, 10)
        train_y = torch.randint(0, 2, (100,))
        val_x = torch.randn(20, 10)
        val_y = torch.randint(0, 2, (20,))
        dm = DataModule(train_x, train_y, val_x, val_y, batch_size=8)
        dm.setup()

        loader = dm.val_dataloader()
        assert loader is not None
        batch = next(iter(loader))
        assert len(batch) == 2

    def test_inputs_converted_to_float(self) -> None:
        train_x = torch.randn(10, 5).double()  # float64
        train_y = torch.randint(0, 2, (10,))
        dm = DataModule(train_x, train_y)
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        assert batch[0].dtype == torch.float32


class TestStratifiedSplit:
    """Tests for stratified_split function."""

    def test_preserves_class_distribution(self) -> None:
        data = torch.randn(100, 10)
        labels = torch.repeat_interleave(torch.arange(4), 25)

        train_data, train_labels, test_data, test_labels = stratified_split(
            data, labels, train_ratio=0.8
        )

        train_dist = torch.bincount(train_labels) / len(train_labels)
        test_dist = torch.bincount(test_labels) / len(test_labels)
        assert torch.allclose(train_dist, test_dist, atol=0.1)

    def test_respects_train_ratio(self) -> None:
        data = torch.randn(100, 10)
        labels = torch.repeat_interleave(torch.arange(4), 25)

        train_data, train_labels, test_data, test_labels = stratified_split(
            data, labels, train_ratio=0.8
        )

        total = len(train_labels) + len(test_labels)
        ratio = len(train_labels) / total
        assert 0.75 <= ratio <= 0.85

    def test_reproducible_with_seed(self) -> None:
        data = torch.randn(100, 10)
        labels = torch.repeat_interleave(torch.arange(4), 25)

        result1 = stratified_split(data, labels, seed=42)
        result2 = stratified_split(data, labels, seed=42)

        assert torch.equal(result1[1], result2[1])

    def test_different_seeds_produce_different_splits(self) -> None:
        data = torch.randn(100, 10)
        labels = torch.repeat_interleave(torch.arange(4), 25)

        result1 = stratified_split(data, labels, seed=42)
        result2 = stratified_split(data, labels, seed=123)

        assert not torch.equal(result1[0], result2[0])


class TestOptionsTrainerKwargs:
    """Tests for Options.trainer_kwargs method."""

    def test_fp32_precision(self) -> None:
        options = Options(precision=Precision.FP32)
        kwargs = options.trainer_kwargs()
        assert kwargs["precision"] == "32-true"

    def test_bf16_precision(self) -> None:
        options = Options(precision=Precision.BF16)
        kwargs = options.trainer_kwargs()
        assert kwargs["precision"] == "bf16-mixed"

    def test_fp16_precision(self) -> None:
        options = Options(precision=Precision.FP16)
        kwargs = options.trainer_kwargs()
        assert kwargs["precision"] == "16-mixed"

    def test_compile_off(self) -> None:
        options = Options.for_benchmark()
        kwargs = options.trainer_kwargs()
        assert "torch_compile" not in kwargs

    def test_compile_on(self) -> None:
        options = Options.for_production()
        kwargs = options.trainer_kwargs()
        assert kwargs.get("torch_compile") is True


class TestCreateTrainer:
    """Tests for create_trainer function."""

    def test_creates_trainer(self) -> None:
        trainer = create_trainer(project="test", logger=False)
        assert isinstance(trainer, pl.Trainer)

    def test_early_stopping_callback(self) -> None:
        trainer = create_trainer(project="test", early_stopping=True, logger=False)
        callback_types = [type(cb).__name__ for cb in trainer.callbacks]
        assert "EarlyStopping" in callback_types

    def test_no_early_stopping_when_disabled(self) -> None:
        trainer = create_trainer(project="test", early_stopping=False, logger=False)
        callback_types = [type(cb).__name__ for cb in trainer.callbacks]
        assert "EarlyStopping" not in callback_types

    def test_checkpoint_callback_when_dir_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = create_trainer(
                project="test", checkpoint_dir=tmpdir, logger=False
            )
            callback_types = [type(cb).__name__ for cb in trainer.callbacks]
            assert "ModelCheckpoint" in callback_types

    def test_lr_monitor_only_with_logger(self) -> None:
        # No LR monitor when logger=False
        trainer = create_trainer(project="test", logger=False)
        callback_types = [type(cb).__name__ for cb in trainer.callbacks]
        assert "LearningRateMonitor" not in callback_types


class TestIntegration:
    """Integration tests for full training workflow."""

    def test_classifier_training(self) -> None:
        # Prepare data
        train_x = torch.randn(80, 10)
        train_y = torch.randint(0, 2, (80,))
        val_x = torch.randn(20, 10)
        val_y = torch.randint(0, 2, (20,))

        dm = DataModule(train_x, train_y, val_x, val_y, batch_size=16)
        model = ConcreteClassifier()

        trainer = create_trainer(
            project="test",
            max_epochs=2,
            early_stopping=False,
            enable_progress_bar=False,
            logger=False,
        )

        trainer.fit(model, dm)
        assert trainer.current_epoch == 2

    def test_trainable_module_training(self) -> None:
        # Regression task
        train_x = torch.randn(80, 10)
        train_y = torch.randn(80, 2)
        val_x = torch.randn(20, 10)
        val_y = torch.randn(20, 2)

        dm = DataModule(train_x, train_y, val_x, val_y, batch_size=16)
        model = ConcreteBaseModule()

        trainer = create_trainer(
            project="test",
            max_epochs=2,
            early_stopping=False,
            enable_progress_bar=False,
            logger=False,
        )

        trainer.fit(model, dm)
        assert trainer.current_epoch == 2
