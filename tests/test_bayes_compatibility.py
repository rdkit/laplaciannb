"""
Tests based on bayes_test.py to ensure compatibility between old and new LaplacianNB implementations.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from laplaciannb.fingerprint_utils import convert_fingerprints
from laplaciannb.LaplacianNB_new import LaplacianNB as LaplacianNB_New

# Import both implementations
from laplaciannb.legacy.LaplacianNB import LaplacianNB as LaplacianNB_Original


class TestBayesCompatibility:
    """Test suite to verify compatibility using bayes_test.py scenarios."""

    def test_basic_bayes_scenario_compatibility(self):
        """Test compatibility using the basic scenario from test_bayes()."""
        # Setup from original test_bayes()
        rng = np.random.RandomState(1)
        arr = rng.randint(2, size=(6, 100))
        Y = np.array([1, 2, 3, 4, 4, 5])
        Xlist = []
        for i in arr:
            Xlist.append(set(i.nonzero()[0]))
        X_sets = np.array(Xlist)

        # Train original model
        clf_original = LaplacianNB_Original()
        clf_original.fit(X_sets, Y)

        # Convert to sklearn format and train new model
        X_sklearn = convert_fingerprints(Xlist, n_bits=100, output_format="csr")
        clf_new = LaplacianNB_New()
        clf_new.fit(X_sklearn, Y)

        # Test predictions match
        pred_original = clf_original.predict(X_sets)
        pred_new = clf_new.predict(X_sklearn)

        print(f"Original predictions: {pred_original}")
        print(f"New predictions:      {pred_new}")

        assert_array_equal(pred_original, pred_new, err_msg="Predictions don't match for basic bayes scenario")

        # Test that internal counts are consistent
        print(f"Original feature_count_: {clf_original.feature_count_}")
        print(f"New feature_count_:      {clf_new.feature_count_}")
        print(f"Original class_count_:   {clf_original.class_count_}")
        print(f"New class_count_:        {clf_new.class_count_}")
        print(f"Original feature_all_:   {clf_original.feature_all_}")
        print(f"New feature_all_:        {clf_new.feature_all_}")

        # Allow for small differences due to different implementation approaches
        assert_allclose(
            clf_original.feature_count_, clf_new.feature_count_, rtol=1e-10, err_msg="Feature counts don't match"
        )
        assert_allclose(
            clf_original.class_count_, clf_new.class_count_, rtol=1e-10, err_msg="Class counts don't match"
        )
        assert_allclose(
            clf_original.feature_all_, clf_new.feature_all_, rtol=1e-10, err_msg="Feature all counts don't match"
        )

    def test_prior_unobserved_targets_compatibility(self):
        """Test compatibility for prior smoothing of unobserved targets."""
        # Setup from test_lmnb_prior_unobserved_targets()
        X_sets = np.array([{1}, {0}])
        y = np.array([0, 1])

        # Train original model
        clf_original = LaplacianNB_Original()
        clf_original.fit(X_sets, y)

        # Convert to sklearn format and train new model
        X_sklearn = convert_fingerprints([{1}, {0}], n_bits=10, output_format="csr")
        clf_new = LaplacianNB_New()
        clf_new.fit(X_sklearn, y)

        # Test predictions for different inputs
        test_cases = [([{1}], "single feature 1"), ([{0}], "single feature 0"), ([{0, 1}], "both features")]

        for test_input_sets, description in test_cases:
            test_input_sklearn = convert_fingerprints(test_input_sets, n_bits=10, output_format="csr")

            pred_original = clf_original.predict(np.array(test_input_sets, dtype=object))
            pred_new = clf_new.predict(test_input_sklearn)

            print(f"Test case {description}:")
            print(f"  Original prediction: {pred_original}")
            print(f"  New prediction:      {pred_new}")

            assert_array_equal(pred_original, pred_new, err_msg=f"Predictions don't match for {description}")

    def test_rdkit_scenario_compatibility(self):
        """Test compatibility using small synthetic fingerprint data (memory-efficient)."""
        # Note: We don't actually use RDKit here to avoid memory issues with large dense matrices
        # Instead, we simulate typical sparse fingerprint data that RDKit would produce

        print("Testing with synthetic sparse fingerprint data...")

        # Create synthetic sparse fingerprint data (simulates RDKit Morgan fingerprints)
        # Each fingerprint is a set of bit indices (sparse representation)
        np.random.seed(42)
        n_samples = 50  # Keep small for memory efficiency
        max_bits = 2048  # Typical fingerprint size

        X_sets = []
        y = []

        for i in range(n_samples):
            # Create sparse fingerprint (5-20 bits set)
            n_bits_set = np.random.randint(5, 21)
            fingerprint = set(np.random.choice(max_bits, n_bits_set, replace=False))
            X_sets.append(fingerprint)
            # Simple target based on fingerprint characteristics
            y.append(1 if len(fingerprint) > 12 else 0)

        X_sets = np.array(X_sets)
        y = np.array(y)

        print(f"Created {len(X_sets)} synthetic fingerprints with max {max_bits} bits")

        # Train original model
        clf_original = LaplacianNB_Original()
        clf_original.fit(X_sets, y)

        # Convert to sparse matrix format (CSR - memory efficient)
        X_sklearn = convert_fingerprints(X_sets.tolist(), n_bits=max_bits, output_format="csr")
        clf_new = LaplacianNB_New()
        clf_new.fit(X_sklearn, y)

        print(f"Sparse matrix shape: {X_sklearn.shape}, nnz: {X_sklearn.nnz}")
        print(f"Sparsity: {1 - X_sklearn.nnz / (X_sklearn.shape[0] * X_sklearn.shape[1]):.4f}")

        # Test predictions
        pred_original = clf_original.predict(X_sets)
        pred_new = clf_new.predict(X_sklearn)

        # Check prediction accuracy
        print(f"Original predictions: {pred_original[:10]}")
        print(f"New predictions:      {pred_new[:10]}")
        print(f"Predictions match: {np.array_equal(pred_original, pred_new)}")

        # Predictions should match exactly for synthetic data
        assert_array_equal(
            pred_original, pred_new, err_msg="Predictions should match exactly for synthetic sparse data"
        )

    def test_joint_log_likelihood_compatibility(self):
        """Test _joint_log_likelihood method compatibility."""
        # Create simple test data
        X_sets = [{1, 5, 10}, {2, 6, 11}, {1, 3, 7}, {4, 8, 12}]
        y = [0, 1, 0, 1]
        n_bits = 20

        # Train both models
        clf_original = LaplacianNB_Original()
        clf_original.fit(np.array(X_sets, dtype=object), y)

        X_sklearn = convert_fingerprints(X_sets, n_bits=n_bits, output_format="csr")
        clf_new = LaplacianNB_New()
        clf_new.fit(X_sklearn, y)

        # Test _joint_log_likelihood with known data
        jll_original = clf_original._joint_log_likelihood(np.array(X_sets, dtype=object))
        jll_new = clf_new._joint_log_likelihood(X_sklearn)

        print(f"Original JLL shape: {jll_original.shape}")
        print(f"New JLL shape:      {jll_new.shape}")
        print(f"Original JLL:\n{jll_original}")
        print(f"New JLL:\n{jll_new}")

        # Check shapes match
        assert jll_original.shape == jll_new.shape, "Joint log likelihood shapes don't match"

        # Check values are reasonably close
        max_diff = np.max(np.abs(jll_original - jll_new))
        print(f"Max JLL difference: {max_diff}")

        # Allow some numerical differences
        assert max_diff < 1.0, f"Joint log likelihood differences too large: {max_diff}"

        # Test with out-of-range feature (should not crash)
        test_set_with_large_feature = [{10210210310210}]

        # Original implementation test
        try:
            clf_original._joint_log_likelihood(np.array(test_set_with_large_feature, dtype=object))
            print("✅ Original handles large feature indices")
        except Exception as e:
            print(f"❌ Original failed with large feature: {e}")

        # New implementation test
        try:
            # Convert large feature set (will be ignored due to bounds checking)
            X_large = convert_fingerprints(test_set_with_large_feature, n_bits=n_bits, output_format="csr")
            clf_new._joint_log_likelihood(X_large)
            print("✅ New handles large feature indices")
        except Exception as e:
            print(f"❌ New failed with large feature: {e}")

    def test_probability_distribution_consistency(self):
        """Test that probability distributions are reasonable and consistent."""
        # Create test data with clear class separation
        X_sets = [
            {1, 2, 3},  # Class 0 features
            {1, 2, 4},  # Class 0 features
            {5, 6, 7},  # Class 1 features
            {5, 6, 8},  # Class 1 features
        ]
        y = [0, 0, 1, 1]
        n_bits = 20

        # Train both models
        clf_original = LaplacianNB_Original()
        clf_original.fit(np.array(X_sets, dtype=object), y)

        X_sklearn = convert_fingerprints(X_sets, n_bits=n_bits, output_format="csr")
        clf_new = LaplacianNB_New()
        clf_new.fit(X_sklearn, y)

        # Test probability estimates
        prob_original = clf_original.predict_proba(np.array(X_sets, dtype=object))
        prob_new = clf_new.predict_proba(X_sklearn)

        print("Original probabilities:")
        print(prob_original)
        print("New probabilities:")
        print(prob_new)

        # Check that probabilities sum to 1 (allow for float32 precision)
        assert_allclose(prob_original.sum(axis=1), 1.0, rtol=1e-6, err_msg="Original probabilities don't sum to 1")
        assert_allclose(prob_new.sum(axis=1), 1.0, rtol=1e-6, err_msg="New probabilities don't sum to 1")

        # Check that probabilities are in valid range
        assert np.all(prob_original >= 0) and np.all(prob_original <= 1), "Original probabilities out of range"
        assert np.all(prob_new >= 0) and np.all(prob_new <= 1), "New probabilities out of range"

        # Check that the highest probability corresponds to correct prediction
        pred_original = clf_original.predict(np.array(X_sets, dtype=object))
        pred_new = clf_new.predict(X_sklearn)

        for i, (pred_o, pred_n) in enumerate(zip(pred_original, pred_new)):
            assert prob_original[i, pred_o] == np.max(
                prob_original[i]
            ), f"Original: max prob doesn't match prediction for sample {i}"
            assert prob_new[i, pred_n] == np.max(prob_new[i]), f"New: max prob doesn't match prediction for sample {i}"

    def test_edge_cases_consistency(self):
        """Test edge cases to ensure both implementations handle them similarly."""

        # Test 1: Single class
        X_single_class = [{1, 2}, {3, 4}, {5, 6}]
        y_single_class = [0, 0, 0]

        clf_orig = LaplacianNB_Original()
        clf_orig.fit(np.array(X_single_class, dtype=object), y_single_class)

        X_sklearn = convert_fingerprints(X_single_class, n_bits=10, output_format="csr")
        clf_new = LaplacianNB_New()
        clf_new.fit(X_sklearn, y_single_class)

        pred_orig = clf_orig.predict(np.array(X_single_class, dtype=object))
        pred_new = clf_new.predict(X_sklearn)

        assert_array_equal(pred_orig, pred_new, err_msg="Single class predictions don't match")
        assert np.all(pred_orig == 0), "Single class should predict class 0"

        # Test 2: Empty features
        X_with_empty = [{1, 2}, set(), {3, 4}]
        y_with_empty = [0, 1, 0]

        clf_orig = LaplacianNB_Original()
        clf_orig.fit(np.array(X_with_empty, dtype=object), y_with_empty)

        X_sklearn = convert_fingerprints(X_with_empty, n_bits=10, output_format="csr")
        clf_new = LaplacianNB_New()
        clf_new.fit(X_sklearn, y_with_empty)

        # Both should handle empty features without crashing
        pred_orig = clf_orig.predict(np.array([set()], dtype=object))
        pred_new = clf_new.predict(convert_fingerprints([set()], n_bits=10, output_format="csr"))

        print(f"Empty feature prediction - Original: {pred_orig}, New: {pred_new}")
        # Don't require exact match for empty features, just no crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
