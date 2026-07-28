"""Tests for execution options module."""

from ml_utils.runs.options import (
    CompileMode,
    Options,
    OptionsError,
    OptionsRationale,
    Precision,
)


class TestOptions:
    """Tests for Options."""

    def test_default_options(self) -> None:
        """Default options have compile=off, fp32, no cuda graphs."""
        options = Options()
        assert options.compile == CompileMode.OFF
        assert options.precision == Precision.FP32
        assert options.cuda_graphs is False

    def test_for_benchmark_factory(self) -> None:
        """for_benchmark returns valid benchmark options."""
        options = Options.for_benchmark()
        assert options.compile == CompileMode.OFF
        assert options.precision == Precision.FP32
        assert options.cuda_graphs is False
        assert options.compile_rationale is not None
        assert "latency" in options.compile_rationale.reason.lower()

    def test_for_production_factory(self) -> None:
        """for_production returns optimized options."""
        options = Options.for_production()
        assert options.compile == CompileMode.MAX_AUTOTUNE
        assert options.precision == Precision.BF16
        assert options.cuda_graphs is True
        assert options.compile_rationale is not None
        assert options.precision_rationale is not None
        assert options.cuda_graphs_rationale is not None

    def test_for_training_factory(self) -> None:
        """for_training returns balanced options."""
        options = Options.for_training()
        assert options.compile == CompileMode.REDUCE_OVERHEAD
        assert options.precision == Precision.BF16
        assert options.cuda_graphs is False


class TestValidation:
    """Tests for options validation."""

    def test_benchmark_accepts_valid_options(self) -> None:
        """Benchmark validation passes for benchmark options."""
        options = Options.for_benchmark()
        result = options.validate_for_benchmark()
        assert result.is_ok()

    def test_benchmark_rejects_compiled_options(self) -> None:
        """Benchmark validation fails when compile is enabled."""
        options = Options(compile=CompileMode.MAX_AUTOTUNE)
        result = options.validate_for_benchmark()
        assert result.is_error()
        err = result.error
        assert isinstance(err, OptionsError)
        assert err.field == "compile"
        assert err.expected == "off"
        assert err.actual == "max-autotune"

    def test_benchmark_rejects_cuda_graphs(self) -> None:
        """Benchmark validation fails when cuda_graphs is enabled."""
        options = Options(cuda_graphs=True)
        result = options.validate_for_benchmark()
        assert result.is_error()
        err = result.error
        assert err.field == "cuda_graphs"

    def test_production_options_fail_benchmark_validation(self) -> None:
        """Production options should fail benchmark validation."""
        options = Options.for_production()
        result = options.validate_for_benchmark()
        assert result.is_error()


class TestOptionsRationale:
    """Tests for OptionsRationale."""

    def test_rationale_with_citation(self) -> None:
        """Rationale can include citation."""
        rationale = OptionsRationale(
            choice="off",
            reason="Test reason",
            citation="Axiom 1",
        )
        assert rationale.citation == "Axiom 1"

    def test_rationale_without_citation(self) -> None:
        """Rationale citation is optional."""
        rationale = OptionsRationale(
            choice="off",
            reason="Test reason",
        )
        assert rationale.citation is None


class TestOptionsError:
    """Tests for OptionsError."""

    def test_error_with_remediation(self) -> None:
        """Error can include remediation suggestion."""
        error = OptionsError(
            field="compile",
            expected="off",
            actual="max-autotune",
            remediation="Use Options.for_benchmark()",
        )
        assert error.remediation is not None

    def test_error_without_remediation(self) -> None:
        """Error remediation is optional."""
        error = OptionsError(
            field="compile",
            expected="off",
            actual="max-autotune",
        )
        assert error.remediation is None
