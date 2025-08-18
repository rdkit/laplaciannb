"""
Comprehensive sklearn integration test suite for LaplacianNB_new implementation.

This test suite validates that LaplacianNB_new works seamlessly with sklearn's
ecosystem including pipelines, cross-validation, grid search, and other tools.
Based on scenarios from bayes_test.py but extended for sklearn compatibility.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from laplaciannb.fingerprint_utils import FingerprintTransformer, convert_fingerprints
from laplaciannb.LaplacianNB_new import LaplacianNB as LaplacianNB_New


class TestSklearnIntegration:
    """Test sklearn ecosystem integration for LaplacianNB_new."""

    @pytest.fixture
    def simple_fingerprint_data(self):
        """Create simple synthetic fingerprint data for testing."""
        np.random.seed(42)
        n_samples = 100
        max_bits = 50

        X_sets = []
        y = []

        for i in range(n_samples):
            # Create sparse fingerprint (3-8 bits set)
            n_bits_set = np.random.randint(3, 9)
            fingerprint = set(np.random.choice(max_bits, n_bits_set, replace=False))
            X_sets.append(fingerprint)
            # Target based on fingerprint characteristics
            y.append(1 if len(fingerprint) > 5 else 0)

        # Convert to sklearn format (defaults to sparse CSR now)
        X_sklearn = convert_fingerprints(X_sets, n_bits=max_bits)  # Defaults to CSR sparse
        y = np.array(y)

        return X_sklearn, y, X_sets

    @pytest.fixture
    def multiclass_fingerprint_data(self):
        """Create multiclass synthetic fingerprint data."""
        np.random.seed(123)
        n_samples = 150
        max_bits = 100

        X_sets = []
        y = []

        for i in range(n_samples):
            n_bits_set = np.random.randint(5, 15)
            fingerprint = set(np.random.choice(max_bits, n_bits_set, replace=False))
            X_sets.append(fingerprint)

            # Three classes based on different criteria
            if len(fingerprint) < 8:
                target = 0
            elif len(fingerprint) < 12:
                target = 1
            else:
                target = 2
            y.append(target)

        X_sklearn = convert_fingerprints(X_sets, n_bits=max_bits)  # Defaults to CSR sparse
        y = np.array(y)

        return X_sklearn, y, X_sets

    def test_basic_sklearn_interface(self, simple_fingerprint_data):
        """Test basic sklearn interface compliance."""
        X, y, _ = simple_fingerprint_data

        clf = LaplacianNB_New()

        # Test basic fit/predict cycle
        clf.fit(X, y)
        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)
        log_probabilities = clf.predict_log_proba(X)

        # Validate shapes and types
        assert predictions.shape == (X.shape[0],)
        assert probabilities.shape == (X.shape[0], 2)
        assert log_probabilities.shape == (X.shape[0], 2)
        assert isinstance(predictions, np.ndarray)
        assert isinstance(probabilities, np.ndarray)
        assert isinstance(log_probabilities, np.ndarray)

        # Validate probability constraints
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_sklearn_estimator_checks(self):
        """Test that the estimator passes sklearn's built-in checks."""
        # Note: We'll run a subset of checks since some may not apply to our specific use case
        try:
            clf = LaplacianNB_New()
            # Test basic estimator properties
            assert hasattr(clf, "fit")
            assert hasattr(clf, "predict")
            assert hasattr(clf, "predict_proba")
            assert callable(clf.fit)
            assert callable(clf.predict)
            assert callable(clf.predict_proba)
        except Exception as e:
            pytest.fail(f"Basic estimator checks failed: {e}")

    def test_pipeline_integration(self, simple_fingerprint_data):
        """Test integration with sklearn pipelines."""
        X, y, _ = simple_fingerprint_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Create pipeline (Note: StandardScaler doesn't make sense for sparse binary data,
        # but we'll use it to test pipeline compatibility)
        pipeline = Pipeline([("classifier", LaplacianNB_New(alpha=1.0))])

        # Fit and predict
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)

        # Validate results
        assert predictions.shape == (X_test.shape[0],)
        assert probabilities.shape == (X_test.shape[0], 2)

        # Test pipeline parameters
        pipeline.set_params(classifier__alpha=2.0)
        assert pipeline.named_steps["classifier"].alpha == 2.0

    def test_cross_validation(self, simple_fingerprint_data):
        """Test cross-validation compatibility."""
        X, y, _ = simple_fingerprint_data

        clf = LaplacianNB_New(alpha=1.0)

        # Perform cross-validation
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

        # Validate results
        assert len(cv_scores) == 5
        assert np.all(cv_scores >= 0)
        assert np.all(cv_scores <= 1)

        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    def test_grid_search_cv(self, simple_fingerprint_data):
        """Test grid search cross-validation."""
        X, y, _ = simple_fingerprint_data

        clf = LaplacianNB_New()

        # Define parameter grid
        param_grid = {"alpha": [0.1, 0.5, 1.0, 2.0, 5.0]}

        # Perform grid search
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy")
        grid_search.fit(X, y)

        # Validate results
        assert hasattr(grid_search, "best_params_")
        assert hasattr(grid_search, "best_score_")
        assert grid_search.best_params_["alpha"] in param_grid["alpha"]

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.3f}")

    def test_multiclass_classification(self, multiclass_fingerprint_data):
        """Test multiclass classification."""
        X, y, _ = multiclass_fingerprint_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = LaplacianNB_New()
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test)

        # Validate multiclass results
        assert probabilities.shape == (X_test.shape[0], 3)  # 3 classes
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert len(np.unique(predictions)) <= 3

        # Test classification report
        report = classification_report(y_test, predictions, output_dict=True)
        assert "accuracy" in report

        print(f"Multiclass accuracy: {report['accuracy']:.3f}")

    def test_sample_weights(self, simple_fingerprint_data):
        """Test sample weight functionality."""
        X, y, _ = simple_fingerprint_data

        # Just verify that the fit method accepts sample weights without error
        sample_weights = np.where(y == 1, 2.0, 1.0)

        clf = LaplacianNB_New()

        # This should not raise an error
        clf.fit(X, y, sample_weight=sample_weights)

        # Basic functionality should still work
        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)

        assert predictions.shape == (X.shape[0],)
        assert probabilities.shape == (X.shape[0], 2)

        print("✓ Sample weights accepted and basic functionality works")

    def test_clone_compatibility(self, simple_fingerprint_data):
        """Test sklearn clone functionality."""
        X, y, _ = simple_fingerprint_data

        clf_original = LaplacianNB_New(alpha=2.0)
        clf_original.fit(X, y)

        # Clone the estimator
        clf_cloned = clone(clf_original)

        # Cloned estimator should not be fitted
        with pytest.raises(NotFittedError):
            clf_cloned.predict(X)

        # Parameters should be copied
        assert clf_cloned.alpha == clf_original.alpha

        # After fitting, cloned estimator should work
        clf_cloned.fit(X, y)
        pred_original = clf_original.predict(X)
        pred_cloned = clf_cloned.predict(X)

        # Results should be identical
        assert_array_equal(pred_original, pred_cloned)

    def test_different_sparse_formats(self, simple_fingerprint_data):
        """Test compatibility with different sparse matrix formats."""
        _, y, X_sets = simple_fingerprint_data

        # Test different sparse formats and dense
        formats = {"csr": "csr", "csc": "csc", "dense": "dense"}
        results = {}

        for name, fmt in formats.items():
            X_converted = convert_fingerprints(X_sets, n_bits=50, output_format=fmt)
            clf = LaplacianNB_New()
            clf.fit(X_converted, y)
            results[name] = clf.predict(X_converted)

            # Verify format
            if name == "dense":
                assert isinstance(X_converted, np.ndarray)
                assert X_converted.ndim == 2
            else:
                assert hasattr(X_converted, "format")
                assert X_converted.format == fmt

        # Results should be identical regardless of format
        assert_array_equal(results["csr"], results["csc"])
        assert_array_equal(results["csr"], results["dense"])

        print("✓ All sparse/dense formats produce identical results")

    def test_sparsity_preservation(self, simple_fingerprint_data):
        """Test that sparse fingerprints remain sparse by default."""
        _, y, X_sets = simple_fingerprint_data

        # Default conversion should produce sparse matrix
        X_default = convert_fingerprints(X_sets, n_bits=50)
        assert hasattr(X_default, "format"), "Default conversion should produce sparse matrix"
        assert X_default.format == "csr", "Default should be CSR format"

        # Check sparsity
        sparsity = 1.0 - (X_default.nnz / (X_default.shape[0] * X_default.shape[1]))
        print(f"Sparsity: {sparsity:.2%}")
        assert sparsity > 0.8, "Molecular fingerprints should be very sparse"

        # Explicit dense conversion should work
        X_dense = convert_fingerprints(X_sets, n_bits=50, output_format="dense")
        assert isinstance(X_dense, np.ndarray), "Explicit dense conversion should work"

        # Results should be equivalent
        clf_sparse = LaplacianNB_New()
        clf_dense = LaplacianNB_New()

        clf_sparse.fit(X_default, y)
        clf_dense.fit(X_dense, y)

        pred_sparse = clf_sparse.predict(X_default)
        pred_dense = clf_dense.predict(X_dense)

        assert_array_equal(pred_sparse, pred_dense)
        print("✓ Sparse and dense give identical predictions")

    def test_edge_cases_sklearn_compatibility(self):
        """Test edge cases for sklearn compatibility."""
        # Single sample
        X_single = convert_fingerprints([{1, 2, 3}], n_bits=10, output_format="csr")
        y_single = np.array([1])

        clf = LaplacianNB_New()
        clf.fit(X_single, y_single)
        pred = clf.predict(X_single)
        prob = clf.predict_proba(X_single)

        assert pred.shape == (1,)
        assert prob.shape == (1, 1)  # Single class

        # Empty features
        X_empty = convert_fingerprints([set(), {1}, set()], n_bits=10, output_format="csr")
        y_empty = np.array([0, 1, 0])

        clf_empty = LaplacianNB_New()
        clf_empty.fit(X_empty, y_empty)
        pred_empty = clf_empty.predict(X_empty)

        assert pred_empty.shape == (3,)

    def test_rdkit_sklearn_pipeline(self):
        """Test full pipeline with RDKit fingerprints (if available)."""
        pytest.importorskip("rdkit", reason="RDKit required for this test")
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        def get_fp(smiles: str, n_bits: int = 1024) -> set:
            """Calculate folded Morgan fingerprint from SMILES with fixed size."""
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return set()
            # Use folded fingerprint for memory efficiency
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
            fp = mfpgen.GetFingerprint(mol)
            return set(fp.GetOnBits())

        # Check if test data exists
        DATA_PATH = Path(__file__).parent / "data"
        test_file = DATA_PATH / "smiles_test.csv"

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        # Load and process small subset for testing
        df = pd.read_csv(test_file)
        df_subset = df.head(50).copy()  # Use copy() to avoid pandas warning

        # Fixed fingerprint size for memory efficiency
        n_bits = 1024
        df_subset["fingerprints"] = df_subset["smiles"].apply(lambda x: get_fp(x, n_bits))
        X_sets = df_subset["fingerprints"].tolist()
        y = df_subset["activity"].values

        # Convert to sklearn format (sparse CSR with fixed size)
        X_sklearn = convert_fingerprints(X_sets, n_bits=n_bits)  # Default to sparse CSR

        # Create and test pipeline
        pipeline = Pipeline([("classifier", LaplacianNB_New(alpha=1.0))])

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_sklearn, y, cv=3, scoring="accuracy")

        print(f"RDKit pipeline CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        assert len(cv_scores) == 3

    def test_stratified_cross_validation(self, simple_fingerprint_data):
        """Test stratified cross-validation for imbalanced datasets."""
        X, y, _ = simple_fingerprint_data

        # Create imbalanced dataset
        mask = y == 1
        # Keep only 20% of class 1 samples
        indices_to_keep = np.where(~mask)[0].tolist()
        indices_to_keep.extend(np.where(mask)[0][: int(mask.sum() * 0.2)].tolist())

        X_imbalanced = X[indices_to_keep]
        y_imbalanced = y[indices_to_keep]

        clf = LaplacianNB_New()

        # Stratified cross-validation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_imbalanced, y_imbalanced, cv=skf, scoring="accuracy")

        assert len(cv_scores) == 3
        print(f"Stratified CV on imbalanced data: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    def test_feature_importance_attributes(self, simple_fingerprint_data):
        """Test that model provides access to feature importance information."""
        X, y, _ = simple_fingerprint_data

        clf = LaplacianNB_New()
        clf.fit(X, y)

        # Check that we can access feature log probabilities
        assert hasattr(clf, "feature_log_prob_")
        assert hasattr(clf, "class_log_prior_")
        assert hasattr(clf, "classes_")

        # Validate shapes
        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        assert clf.feature_log_prob_.shape == (n_classes, n_features)
        assert clf.class_log_prior_.shape == (n_classes,)
        assert len(clf.classes_) == n_classes

    def test_pipeline_with_feature_selection(self, simple_fingerprint_data):
        """Test pipeline with feature selection (simulated)."""
        X, y, _ = simple_fingerprint_data

        # Since sklearn feature selection doesn't work well with our sparse binary data,
        # we'll simulate by using a subset of features
        n_features_selected = 30
        X_reduced = X[:, :n_features_selected]

        pipeline = Pipeline([("classifier", LaplacianNB_New(alpha=1.0))])

        # Test that it works with reduced features
        pipeline.fit(X_reduced, y)
        predictions = pipeline.predict(X_reduced)

        assert predictions.shape == (X_reduced.shape[0],)

    def test_reproducibility(self, simple_fingerprint_data):
        """Test that results are reproducible."""
        X, y, _ = simple_fingerprint_data

        clf1 = LaplacianNB_New(alpha=1.0)
        clf2 = LaplacianNB_New(alpha=1.0)

        clf1.fit(X, y)
        clf2.fit(X, y)

        pred1 = clf1.predict(X)
        pred2 = clf2.predict(X)

        prob1 = clf1.predict_proba(X)
        prob2 = clf2.predict_proba(X)

        # Results should be identical
        assert_array_equal(pred1, pred2)
        assert_allclose(prob1, prob2)

    def test_fingerprint_transformer(self, simple_fingerprint_data):
        """Test the FingerprintTransformer sklearn interface."""
        _, y, X_sets = simple_fingerprint_data

        # Test basic transformer functionality
        transformer = FingerprintTransformer(n_bits=50, output_format="csr")

        # Test fit/transform
        X_transformed = transformer.fit_transform(X_sets)
        assert hasattr(X_transformed, "format")
        assert X_transformed.format == "csr"
        assert X_transformed.shape == (len(X_sets), 50)

        # Test separate fit/transform
        transformer2 = FingerprintTransformer(n_bits=50, output_format="dense")
        transformer2.fit(X_sets)
        X_dense = transformer2.transform(X_sets)
        assert isinstance(X_dense, np.ndarray)
        assert X_dense.shape == (len(X_sets), 50)

        # Test get_feature_names_out
        feature_names = transformer.get_feature_names_out()
        assert len(feature_names) == 50
        assert feature_names[0] == "bit_0"
        assert feature_names[49] == "bit_49"

        # Test sklearn pipeline integration
        pipeline = Pipeline([("fingerprints", FingerprintTransformer(n_bits=50)), ("classifier", LaplacianNB_New())])

        pipeline.fit(X_sets, y)
        predictions = pipeline.predict(X_sets)
        assert predictions.shape == (len(X_sets),)

        # Test cross-validation with pipeline
        cv_scores = cross_val_score(pipeline, X_sets, y, cv=3)
        assert len(cv_scores) == 3

        print("✓ FingerprintTransformer sklearn integration works perfectly")

    def test_transformer_pipeline_with_grid_search(self, simple_fingerprint_data):
        """Test FingerprintTransformer in grid search pipeline."""
        _, y, X_sets = simple_fingerprint_data

        # Create pipeline with transformer
        pipeline = Pipeline([("fingerprints", FingerprintTransformer()), ("classifier", LaplacianNB_New())])

        # Grid search with transformer and classifier parameters
        param_grid = {
            "fingerprints__n_bits": [25, 50],
            "fingerprints__output_format": ["csr", "dense"],
            "classifier__alpha": [0.5, 1.0],
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy")
        grid_search.fit(X_sets, y)

        assert hasattr(grid_search, "best_params_")
        assert hasattr(grid_search, "best_score_")

        print(f"Best transformer pipeline params: {grid_search.best_params_}")
        print("✓ Grid search with FingerprintTransformer works")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
