"""Tests for ml_utils.data module."""

import numpy as np

from ml_utils.data import stratified_indices

_RNG = np.random.default_rng(42)


class TestStratifiedIndices:
    """Tests for stratified_indices function."""

    def test_returns_tuple_of_arrays(self) -> None:
        """stratified_indices returns tuple of train/test index arrays."""
        labels = np.repeat(np.arange(4), 25).astype(np.int64)
        train_idx, test_idx = stratified_indices(labels, test_fraction=0.2)
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)

    def test_indices_cover_all_samples(self) -> None:
        """Train and test indices cover all samples exactly once."""
        labels = np.repeat(np.arange(4), 25).astype(np.int64)
        train_idx, test_idx = stratified_indices(labels, test_fraction=0.2)

        all_indices = np.sort(np.concatenate([train_idx, test_idx]))
        expected = np.arange(len(labels))
        np.testing.assert_array_equal(all_indices, expected)

    def test_no_overlap(self) -> None:
        """Train and test indices don't overlap."""
        labels = np.repeat(np.arange(4), 25).astype(np.int64)
        train_idx, test_idx = stratified_indices(labels, test_fraction=0.2)

        train_set = set(train_idx.tolist())
        test_set = set(test_idx.tolist())
        assert len(train_set & test_set) == 0

    def test_preserves_class_distribution(self) -> None:
        """Train and test maintain similar class distribution."""
        labels = np.repeat(np.arange(4), 50).astype(np.int64)
        train_idx, test_idx = stratified_indices(labels, test_fraction=0.2)

        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        train_dist = np.bincount(train_labels) / len(train_labels)
        test_dist = np.bincount(test_labels) / len(test_labels)

        # Each class should have ~25% in both sets
        np.testing.assert_array_almost_equal(train_dist, test_dist, decimal=1)

    def test_respects_test_fraction(self) -> None:
        """Test set size respects test_fraction parameter."""
        labels = np.repeat(np.arange(4), 100).astype(np.int64)
        train_idx, test_idx = stratified_indices(labels, test_fraction=0.3)

        total = len(train_idx) + len(test_idx)
        actual_fraction = len(test_idx) / total
        assert 0.25 <= actual_fraction <= 0.35

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces same split."""
        labels = np.repeat(np.arange(4), 25).astype(np.int64)

        train1, test1 = stratified_indices(labels, test_fraction=0.2, seed=42)
        train2, test2 = stratified_indices(labels, test_fraction=0.2, seed=42)

        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(test1, test2)

    def test_different_seeds_produce_different_splits(self) -> None:
        """Different seeds produce different splits."""
        labels = np.repeat(np.arange(4), 25).astype(np.int64)

        train1, _ = stratified_indices(labels, test_fraction=0.2, seed=42)
        train2, _ = stratified_indices(labels, test_fraction=0.2, seed=99)

        assert not np.array_equal(train1, train2)

    def test_each_class_has_at_least_one_test_sample(self) -> None:
        """Each class gets at least one sample in test set."""
        # Even with very small test fraction
        labels = np.repeat(np.arange(4), 10).astype(np.int64)
        _, test_idx = stratified_indices(labels, test_fraction=0.1)

        test_labels = labels[test_idx]
        unique_classes = np.unique(test_labels)
        assert len(unique_classes) == 4

    def test_handles_imbalanced_classes(self) -> None:
        """Handles imbalanced class distributions."""
        # Create imbalanced: class 0 has 100, others have 10 each
        labels = np.array([0] * 100 + [1] * 10 + [2] * 10 + [3] * 10, dtype=np.int64)
        train_idx, test_idx = stratified_indices(labels, test_fraction=0.2)

        # All classes should be present in both splits
        train_classes = np.unique(labels[train_idx])
        test_classes = np.unique(labels[test_idx])
        assert len(train_classes) == 4
        assert len(test_classes) == 4
