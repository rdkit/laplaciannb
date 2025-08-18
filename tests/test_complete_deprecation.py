"""
Test the complete deprecation and migration system.
"""

import warnings

import numpy as np
import pytest

from laplaciannb.fingerprint_utils import convert_fingerprints


def test_new_version_detects_set_input():
    """Test that new version detects and rejects legacy set input with helpful error."""
    from laplaciannb import LaplacianNB

    X_sets = [{1, 2, 3}, {4, 5, 6}]
    y = [0, 1]

    clf = LaplacianNB()

    # Should raise ValueError with helpful message
    with pytest.raises(ValueError) as exc_info:
        clf.fit(X_sets, y)

    error_message = str(exc_info.value)
    assert "LEGACY INPUT FORMAT ERROR" in error_message
    assert "convert_fingerprints" in error_message
    assert "laplaciannb.legacy" in error_message


def test_new_version_detects_numpy_array_of_sets():
    """Test detection of numpy array with object dtype containing sets."""
    from laplaciannb import LaplacianNB

    X_sets = np.array([{1, 2, 3}, {4, 5, 6}], dtype=object)
    y = [0, 1]

    clf = LaplacianNB()

    # Should raise ValueError with helpful message
    with pytest.raises(ValueError) as exc_info:
        clf.fit(X_sets, y)

    error_message = str(exc_info.value)
    assert "LEGACY INPUT FORMAT ERROR" in error_message


def test_new_version_detects_predict_method():
    """Test detection during predict method calls."""
    from laplaciannb import LaplacianNB
    from laplaciannb.fingerprint_utils import convert_fingerprints

    # First fit with proper sklearn format
    X_proper = convert_fingerprints([{1, 2}, {3, 4}, {5, 6}], n_bits=10)
    y = [0, 1, 0]
    clf = LaplacianNB()
    clf.fit(X_proper, y)

    # Now try to predict with set format
    X_sets = [{1, 2, 3}]

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        clf.predict(X_sets)

    error_message = str(exc_info.value)
    assert "LEGACY INPUT FORMAT ERROR" in error_message


def test_recommended_migration_path():
    """Test that the recommended migration path works without warnings."""
    from laplaciannb import LaplacianNB

    X_sets = [{1, 2, 3}, {4, 5, 6}, {1, 4, 7}]
    y = [0, 1, 0]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Recommended path: convert fingerprints first
        X = convert_fingerprints(X_sets, n_bits=10)
        clf = LaplacianNB()
        clf.fit(X, y)
        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)

        # Should work without user warnings (only import warnings are OK)
        user_warnings = [warning for warning in w if issubclass(warning.category, UserWarning)]
        assert len(user_warnings) == 0

        # Results should be valid
        assert predictions.shape == (3,)
        assert probabilities.shape == (3, 2)


def test_legacy_version_still_works():
    """Test that legacy version still works for backward compatibility."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress deprecation warnings

        from laplaciannb.legacy import LaplacianNB as LegacyLaplacianNB

        X_sets = np.array([{1, 2, 3}, {4, 5, 6}, {1, 4, 7}], dtype=object)
        y = [0, 1, 0]

        clf = LegacyLaplacianNB()
        clf.fit(X_sets, y)
        predictions = clf.predict(X_sets)

        assert predictions.shape == (3,)


def test_complete_migration_scenario():
    """Test a complete migration from legacy to new."""
    # Step 1: User starts with legacy
    X_sets = [{1, 2, 3}, {4, 5, 6}, {1, 4, 7}]
    y = [0, 1, 0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from laplaciannb.legacy import LaplacianNB as LegacyLaplacianNB

        X_legacy = np.array(X_sets, dtype=object)
        clf_legacy = LegacyLaplacianNB()
        clf_legacy.fit(X_legacy, y)
        pred_legacy = clf_legacy.predict(X_legacy)

    # Step 2: User tries new version with same data (gets helpful error)
    from laplaciannb import LaplacianNB as NewLaplacianNB

    clf_new_wrong = NewLaplacianNB()
    with pytest.raises(ValueError) as exc_info:
        clf_new_wrong.fit(X_sets, y)

    # Should get helpful guidance in error message
    error_message = str(exc_info.value)
    assert "LEGACY INPUT FORMAT ERROR" in error_message
    assert "convert_fingerprints" in error_message

    # Step 3: User follows guidance and migrates successfully
    X_new = convert_fingerprints(X_sets, n_bits=10)
    clf_new = NewLaplacianNB()
    clf_new.fit(X_new, y)
    pred_new = clf_new.predict(X_new)

    # Step 4: Verify identical results
    assert np.array_equal(pred_legacy, pred_new)


def test_single_set_detection():
    """Test detection of single set input (single fingerprint)."""
    from laplaciannb import LaplacianNB

    # User passes a single set instead of list of sets
    X_single_set = {1, 2, 3, 4, 5}
    y = [1]

    clf = LaplacianNB()

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        clf.fit(X_single_set, y)

    error_message = str(exc_info.value)
    assert "LEGACY INPUT FORMAT ERROR" in error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
