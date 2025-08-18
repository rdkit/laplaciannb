"""
Test the main import paths and ensure proper version selection.
"""

import warnings

import pytest

from laplaciannb.fingerprint_utils import convert_fingerprints


def test_main_import_gives_new_version():
    """Test that importing from main module gives the new sklearn-compatible version."""
    from laplaciannb import LaplacianNB

    # Should be the new implementation (sklearn-compatible)
    assert LaplacianNB.__module__ == "laplaciannb.LaplacianNB_new"

    # Should have sklearn-style attributes after fitting
    X_sets = [{1, 2, 3}, {4, 5, 6}, {1, 4, 7}]
    y = [0, 1, 0]
    X = convert_fingerprints(X_sets, n_bits=10)

    clf = LaplacianNB()
    clf.fit(X, y)

    # Should have sklearn-style attributes
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "n_features_in_")
    assert hasattr(clf, "feature_log_prob_")


def test_legacy_import_gives_legacy_version():
    """Test that importing from legacy module gives the legacy version."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress deprecation warnings
        from laplaciannb.legacy import LaplacianNB as LegacyLaplacianNB

    # Should be the legacy implementation
    assert "legacy" in LegacyLaplacianNB.__module__


def test_fingerprint_utils_available():
    """Test that fingerprint utilities are available from main module."""
    from laplaciannb import (
        FingerprintTransformer,
        RDKitFingerprintConverter,
        convert_fingerprints,
        rdkit_sparse_to_csr,
        rdkit_sparse_to_dense,
        rdkit_sparse_to_numpy,
        rdkit_sparse_to_sklearn,
    )

    # All should be callable
    assert callable(FingerprintTransformer)
    assert callable(RDKitFingerprintConverter)
    assert callable(convert_fingerprints)
    assert callable(rdkit_sparse_to_dense)
    assert callable(rdkit_sparse_to_csr)
    assert callable(rdkit_sparse_to_numpy)
    assert callable(rdkit_sparse_to_sklearn)


def test_version_info_available():
    """Test that version information is available."""
    from laplaciannb import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
