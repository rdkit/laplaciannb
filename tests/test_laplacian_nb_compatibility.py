"""
Tests to verify that the refactored LaplacianNB produces the same results
as the original implementation.
"""

import numpy as np
import pytest

# Import the converters
from laplaciannb.fingerprint_utils import convert_fingerprints

# New version with sklearn-compatible input
from laplaciannb.LaplacianNB_new import LaplacianNB as LaplacianNB_New

# Import both versions of LaplacianNB
# Original version with set-based operations
from laplaciannb.legacy.LaplacianNB import LaplacianNB as LaplacianNB_Original


class TestLaplacianNBCompatibility:
    """Test suite to verify compatibility between old and new LaplacianNB implementations."""

    @pytest.fixture
    def generate_fingerprint_data(self):
        """Generate test data in fingerprint format."""
        np.random.seed(42)
        n_samples = 100
        n_bits = 256  # Smaller for faster testing

        # Generate fingerprints as sets (original format)
        X_sets = []
        for _ in range(n_samples):
            n_on_bits = np.random.randint(5, 30)
            on_bits = set(np.random.choice(n_bits, n_on_bits, replace=False))
            X_sets.append(on_bits)

        # Generate labels
        y = np.random.randint(0, 3, n_samples)

        return X_sets, y, n_bits

    def test_same_predictions_binary_classification(self, generate_fingerprint_data):
        """Test that both versions give same predictions for binary classification."""
        X_sets, y_multi, n_bits = generate_fingerprint_data

        # Make binary labels
        y = (y_multi > 0).astype(int)

        # Train original model with sets
        clf_original = LaplacianNB_Original(alpha=1.0)
        clf_original.fit(np.array(X_sets, dtype=object), y)

        # Convert to sklearn format for new model
        X_sklearn = convert_fingerprints(X_sets, n_bits=n_bits, output_format="csr")

        # Train new model
        clf_new = LaplacianNB_New(alpha=1.0)
        clf_new.fit(X_sklearn, y)

        # Make predictions with both models
        pred_original = clf_original.predict(np.array(X_sets[:20], dtype=object))
        pred_new = clf_new.predict(X_sklearn[:20])

        # Assert predictions are the same
        np.testing.assert_array_equal(
            pred_original, pred_new, err_msg="Binary predictions differ between implementations"
        )

    def test_same_predictions_multiclass(self, generate_fingerprint_data):
        """Test that both versions give same predictions for multiclass classification."""
        X_sets, y, n_bits = generate_fingerprint_data

        # Train original model with sets
        clf_original = LaplacianNB_Original(alpha=1.0)
        clf_original.fit(np.array(X_sets, dtype=object), y)

        # Convert to sklearn format for new model
        X_sklearn = convert_fingerprints(X_sets, n_bits=n_bits, output_format="csr")

        # Train new model
        clf_new = LaplacianNB_New(alpha=1.0)
        clf_new.fit(X_sklearn, y)

        # Make predictions with both models
        pred_original = clf_original.predict(np.array(X_sets[:20], dtype=object))
        pred_new = clf_new.predict(X_sklearn[:20])

        # Assert predictions are the same
        np.testing.assert_array_equal(
            pred_original, pred_new, err_msg="Multiclass predictions differ between implementations"
        )

    def test_same_probabilities(self, generate_fingerprint_data):
        """Test that both versions give same probability estimates."""
        X_sets, y, n_bits = generate_fingerprint_data

        # Train original model with sets
        clf_original = LaplacianNB_Original(alpha=1.0)
        clf_original.fit(np.array(X_sets, dtype=object), y)

        # Convert to sklearn format for new model
        X_sklearn = convert_fingerprints(X_sets, n_bits=n_bits, output_format="csr")

        # Train new model
        clf_new = LaplacianNB_New(alpha=1.0)
        clf_new.fit(X_sklearn, y)

        # Get probabilities from both models
        test_samples = 10
        prob_original = clf_original.predict_proba(np.array(X_sets[:test_samples], dtype=object))
        prob_new = clf_new.predict_proba(X_sklearn[:test_samples])

        # Assert probabilities are very close (allowing for floating point differences)
        np.testing.assert_allclose(
            prob_original,
            prob_new,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Probability estimates differ between implementations",
        )

    def test_same_log_probabilities(self, generate_fingerprint_data):
        """Test that both versions give same log probability estimates."""
        X_sets, y, n_bits = generate_fingerprint_data

        # Train original model with sets
        clf_original = LaplacianNB_Original(alpha=1.0)
        clf_original.fit(np.array(X_sets, dtype=object), y)

        # Convert to sklearn format for new model
        X_sklearn = convert_fingerprints(X_sets, n_bits=n_bits, output_format="csr")

        # Train new model
        clf_new = LaplacianNB_New(alpha=1.0)
        clf_new.fit(X_sklearn, y)

        # Get log probabilities from both models
        test_samples = 10
        log_prob_original = clf_original.predict_log_proba(np.array(X_sets[:test_samples], dtype=object))
        log_prob_new = clf_new.predict_log_proba(X_sklearn[:test_samples])

        # Assert log probabilities are very close
        np.testing.assert_allclose(
            log_prob_original,
            log_prob_new,
            rtol=1e-4,
            atol=1e-7,
            err_msg="Log probability estimates differ between implementations",
        )

    def test_different_alpha_values(self, generate_fingerprint_data):
        """Test consistency across different smoothing parameters."""
        X_sets, y, n_bits = generate_fingerprint_data
        X_sklearn = convert_fingerprints(X_sets, n_bits=n_bits, output_format="csr")

        alpha_values = [0.1, 0.5, 1.0, 2.0, 10.0]

        for alpha in alpha_values:
            # Train both models
            clf_original = LaplacianNB_Original(alpha=alpha)
            clf_original.fit(np.array(X_sets, dtype=object), y)

            clf_new = LaplacianNB_New(alpha=alpha)
            clf_new.fit(X_sklearn, y)

            # Compare predictions
            pred_original = clf_original.predict(np.array(X_sets[:20], dtype=object))
            pred_new = clf_new.predict(X_sklearn[:20])

            np.testing.assert_array_equal(pred_original, pred_new, err_msg=f"Predictions differ for alpha={alpha}")

    def test_sample_weights(self, generate_fingerprint_data):
        """Test that both versions handle sample weights the same way."""
        X_sets, y, n_bits = generate_fingerprint_data
        X_sklearn = convert_fingerprints(X_sets, n_bits=n_bits, output_format="csr")

        # Create sample weights
        sample_weight = np.random.rand(len(y))

        # Train both models with sample weights
        clf_original = LaplacianNB_Original(alpha=1.0)
        clf_original.fit(np.array(X_sets, dtype=object), y, sample_weight=sample_weight)

        clf_new = LaplacianNB_New(alpha=1.0)
        clf_new.fit(X_sklearn, y, sample_weight=sample_weight)

        # Compare predictions
        pred_original = clf_original.predict(np.array(X_sets[:20], dtype=object))
        pred_new = clf_new.predict(X_sklearn[:20])

        np.testing.assert_array_equal(pred_original, pred_new, err_msg="Predictions differ when using sample weights")

    def test_sparse_vs_dense_input(self, generate_fingerprint_data):
        """Test that new version gives same results with sparse and dense input."""
        X_sets, y, n_bits = generate_fingerprint_data

        # Convert to both sparse and dense formats
        X_sparse = convert_fingerprints(X_sets, n_bits=n_bits, output_format="csr")
        X_dense = convert_fingerprints(X_sets, n_bits=n_bits, output_format="dense")

        # Train new model with both formats
        clf_sparse = LaplacianNB_New(alpha=1.0)
        clf_sparse.fit(X_sparse, y)

        clf_dense = LaplacianNB_New(alpha=1.0)
        clf_dense.fit(X_dense, y)

        # Compare predictions
        pred_sparse = clf_sparse.predict(X_sparse[:20])
        pred_dense = clf_dense.predict(X_dense[:20])

        np.testing.assert_array_equal(
            pred_sparse, pred_dense, err_msg="Predictions differ between sparse and dense input"
        )

        # Compare probabilities
        prob_sparse = clf_sparse.predict_proba(X_sparse[:20])
        prob_dense = clf_dense.predict_proba(X_dense[:20])

        np.testing.assert_allclose(
            prob_sparse,
            prob_dense,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Probabilities differ between sparse and dense input",
        )

    def test_feature_counting_consistency(self, generate_fingerprint_data):
        """Test that feature counting is consistent between implementations."""
        X_sets, y, n_bits = generate_fingerprint_data
        X_sklearn = convert_fingerprints(X_sets, n_bits=n_bits, output_format="csr")

        # Train both models
        clf_original = LaplacianNB_Original(alpha=1.0)
        clf_original.fit(np.array(X_sets, dtype=object), y)

        clf_new = LaplacianNB_New(alpha=1.0)
        clf_new.fit(X_sklearn, y)

        # Check that feature counts are consistent
        # The original stores counts differently, but total counts should match
        assert clf_original.feature_all_ == clf_new.feature_all_, "Total feature counts differ between implementations"

        # Check class counts
        np.testing.assert_allclose(
            clf_original.class_count_, clf_new.class_count_, err_msg="Class counts differ between implementations"
        )

    def test_single_class_edge_case(self):
        """Test handling of degenerate case with single class."""
        np.random.seed(42)
        n_samples = 20
        n_bits = 128

        # Generate fingerprints
        X_sets = []
        for _ in range(n_samples):
            n_on_bits = np.random.randint(5, 15)
            on_bits = set(np.random.choice(n_bits, n_on_bits, replace=False))
            X_sets.append(on_bits)

        # Single class labels
        y = np.ones(n_samples)

        # Train both models
        clf_original = LaplacianNB_Original(alpha=1.0)
        clf_original.fit(np.array(X_sets, dtype=object), y)

        X_sklearn = convert_fingerprints(X_sets, n_bits=n_bits, output_format="csr")
        clf_new = LaplacianNB_New(alpha=1.0)
        clf_new.fit(X_sklearn, y)

        # Both should predict the same class
        pred_original = clf_original.predict(np.array(X_sets[:5], dtype=object))
        pred_new = clf_new.predict(X_sklearn[:5])

        np.testing.assert_array_equal(pred_original, pred_new, err_msg="Single class predictions differ")

    def test_empty_features_handling(self):
        """Test handling of samples with no active features."""
        n_bits = 128

        # Create samples with some empty fingerprints
        X_sets = [
            {1, 2, 3},
            set(),  # Empty fingerprint
            {5, 10},
            set(),  # Another empty
            {20, 30, 40},
        ]
        y = [0, 0, 1, 1, 1]

        # Train both models
        clf_original = LaplacianNB_Original(alpha=1.0)
        clf_original.fit(np.array(X_sets, dtype=object), y)

        X_sklearn = convert_fingerprints(X_sets, n_bits=n_bits, output_format="csr")
        clf_new = LaplacianNB_New(alpha=1.0)
        clf_new.fit(X_sklearn, y)

        # Make predictions
        pred_original = clf_original.predict(np.array(X_sets, dtype=object))
        pred_new = clf_new.predict(X_sklearn)

        np.testing.assert_array_equal(pred_original, pred_new, err_msg="Predictions differ with empty fingerprints")


def run_compatibility_tests():
    """Run all compatibility tests and report results."""

    print("Running LaplacianNB Compatibility Tests")
    print("=" * 60)

    # Run tests using pytest
    test = TestLaplacianNBCompatibility()

    # Generate test data
    np.random.seed(42)
    n_samples = 100
    n_bits = 256
    X_sets = []
    for _ in range(n_samples):
        n_on_bits = np.random.randint(5, 30)
        on_bits = set(np.random.choice(n_bits, n_on_bits, replace=False))
        X_sets.append(on_bits)
    y = np.random.randint(0, 3, n_samples)

    test_data = (X_sets, y, n_bits)

    tests = [
        ("Binary Classification", test.test_same_predictions_binary_classification),
        ("Multiclass Classification", test.test_same_predictions_multiclass),
        ("Probability Estimates", test.test_same_probabilities),
        ("Log Probability Estimates", test.test_same_log_probabilities),
        ("Different Alpha Values", test.test_different_alpha_values),
        ("Sample Weights", test.test_sample_weights),
        ("Sparse vs Dense Input", test.test_sparse_vs_dense_input),
        ("Feature Counting", test.test_feature_counting_consistency),
        ("Single Class Edge Case", test.test_single_class_edge_case),
        ("Empty Features", test.test_empty_features_handling),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func.__name__ in ["test_single_class_edge_case", "test_empty_features_handling"]:
                test_func()
            else:
                test_func(test_data)
            print(f"✓ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {str(e)}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    return passed, failed


if __name__ == "__main__":
    run_compatibility_tests()
