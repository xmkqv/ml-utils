"""Tests for ml_utils.nn.FocalLoss."""

import torch

from ml_utils.loss import FocalLoss


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_returns_scalar(self) -> None:
        """FocalLoss returns scalar with default reduction."""
        loss_fn = FocalLoss()
        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # scalar

    def test_gamma_zero_equals_cross_entropy(self) -> None:
        """With gamma=0, FocalLoss equals CrossEntropyLoss."""
        focal = FocalLoss(gamma=0.0)
        ce = torch.nn.CrossEntropyLoss()

        torch.manual_seed(42)
        logits = torch.randn(32, 4)
        targets = torch.randint(0, 4, (32,))

        focal_loss = focal(logits, targets)
        ce_loss = ce(logits, targets)

        torch.testing.assert_close(focal_loss, ce_loss, rtol=1e-4, atol=1e-6)

    def test_higher_gamma_focuses_on_hard_examples(self) -> None:
        """Higher gamma gives lower loss for confident predictions."""
        # Create easy predictions (high confidence correct)
        logits_easy = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
        targets = torch.tensor([0])

        low_gamma = FocalLoss(gamma=0.0)
        high_gamma = FocalLoss(gamma=2.0)

        loss_low = low_gamma(logits_easy, targets)
        loss_high = high_gamma(logits_easy, targets)

        # Higher gamma should reduce loss for confident predictions
        assert loss_high < loss_low

    def test_alpha_weighting(self) -> None:
        """Alpha weights classes appropriately."""
        # Create unbalanced alpha
        alpha = torch.tensor([0.1, 0.3, 0.3, 0.3])
        focal = FocalLoss(gamma=0.0, alpha=alpha)

        # Predictions for class 0 (low alpha) vs class 1 (high alpha)
        logits = torch.tensor([[0.0, 10.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]])
        targets_low_alpha = torch.tensor([0, 0])  # class 0, alpha=0.1
        targets_high_alpha = torch.tensor([1, 1])  # class 1, alpha=0.3

        loss_low_alpha = focal(logits, targets_low_alpha)
        loss_high_alpha = focal(logits, targets_high_alpha)

        # Higher alpha should give higher loss contribution
        assert loss_high_alpha > loss_low_alpha

    def test_reduction_none(self) -> None:
        """reduction='none' returns per-sample losses."""
        focal = FocalLoss(reduction="none")
        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))

        loss = focal(logits, targets)
        assert loss.shape == (10,)

    def test_reduction_sum(self) -> None:
        """reduction='sum' returns sum of losses."""
        focal_sum = FocalLoss(reduction="sum")
        focal_none = FocalLoss(reduction="none")

        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))

        sum_loss = focal_sum(logits, targets)
        none_loss = focal_none(logits, targets)

        torch.testing.assert_close(sum_loss, none_loss.sum())

    def test_reduction_mean(self) -> None:
        """reduction='mean' returns mean of losses."""
        focal_mean = FocalLoss(reduction="mean")
        focal_none = FocalLoss(reduction="none")

        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))

        mean_loss = focal_mean(logits, targets)
        none_loss = focal_none(logits, targets)

        torch.testing.assert_close(mean_loss, none_loss.mean())

    def test_gradient_flows(self) -> None:
        """Gradients flow through focal loss."""
        focal = FocalLoss()
        logits = torch.randn(10, 4, requires_grad=True)
        targets = torch.randint(0, 4, (10,))

        loss = focal(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_alpha_as_tensor(self) -> None:
        """Alpha can be provided as tensor."""
        alpha = torch.tensor([0.25, 0.25, 0.25, 0.25])
        focal = FocalLoss(gamma=2.0, alpha=alpha)

        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))

        loss = focal(logits, targets)
        assert loss.dim() == 0

    def test_device_transfer(self) -> None:
        """Alpha moves to correct device with logits."""
        alpha = torch.tensor([0.25, 0.25, 0.25, 0.25])
        focal = FocalLoss(gamma=2.0, alpha=alpha)

        # On CPU (default)
        logits = torch.randn(10, 4)
        targets = torch.randint(0, 4, (10,))

        loss = focal(logits, targets)
        assert loss.device == logits.device
