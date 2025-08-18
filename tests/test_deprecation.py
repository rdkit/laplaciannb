"""
Tests for deprecation warnings and legacy/new version compatibility.
"""

import warnings

import numpy as np
import pytest

from laplaciannb.fingerprint_utils import convert_fingerprints


class TestDeprecationWarnings:
    """Test that deprecation warnings are properly issued."""

    @pytest.mark.skip(reason="Import warnings are only triggered once per session due to Python import caching")
    def test_legacy_module_import_warning(self):
        """Test that importing from legacy module issues deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import from legacy module should trigger warning
            from laplaciannb.legacy import LaplacianNB  # noqa: F401

            # Check that a deprecation warning was issued
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "DEPRECATED legacy LaplacianNB" in str(w[0].message)
            assert "sklearn-compatible" in str(w[0].message)

    def test_legacy_class_instantiation_warning(self):
        """Test that instantiating legacy class issues deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from laplaciannb.legacy import LaplacianNB as LegacyLaplacianNB

            # Clear the import warning to focus on instantiation warning
            w.clear()

            # Instantiate legacy class should trigger additional warning
            LegacyLaplacianNB()

            # Check that a deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "DEPRECATED legacy LaplacianNB" in str(w[0].message)

    def test_new_version_no_warnings(self):
        """Test that new version doesn't issue deprecation warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import and use new version
            from laplaciannb import LaplacianNB

            LaplacianNB()

            # Should not have any deprecation warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

    def test_recommended_import_path(self):
        """Test that recommended import path works without warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This is the recommended way
            from laplaciannb import LaplacianNB

            # Create sample data
            X_sets = [{1, 2, 3}, {4, 5, 6}, {1, 4, 7}]
            y = [0, 1, 0]

            # Convert and use
            X = convert_fingerprints(X_sets, n_bits=10)
            clf = LaplacianNB()
            clf.fit(X, y)
            predictions = clf.predict(X)

            # Should work without deprecation warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0
            assert predictions.shape == (3,)


class TestBothVersionsAvailable:
    """Test that both versions are available and work correctly."""

    def test_both_versions_importable(self):
        """Test that both legacy and new versions can be imported."""
        # New version (recommended)
        from laplaciannb import LaplacianNB as NewLaplacianNB

        # Legacy version (deprecated)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from laplaciannb.legacy import LaplacianNB as LegacyLaplacianNB

        # Both should be classes
        assert callable(NewLaplacianNB)
        assert callable(LegacyLaplacianNB)

    def test_different_implementations(self):
        """Test that legacy and new are different implementations."""
        from laplaciannb import LaplacianNB as NewLaplacianNB

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from laplaciannb.legacy import LaplacianNB as LegacyLaplacianNB

        # They should be different classes
        assert NewLaplacianNB is not LegacyLaplacianNB
        assert NewLaplacianNB.__module__ != LegacyLaplacianNB.__module__

    def test_identical_api_basic_usage(self):
        """Test that both versions have similar basic API."""
        from laplaciannb import LaplacianNB as NewLaplacianNB

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from laplaciannb.legacy import LaplacianNB as LegacyLaplacianNB

        # Both should have the same basic methods
        for method in ["fit", "predict", "predict_proba", "predict_log_proba"]:
            assert hasattr(NewLaplacianNB(), method)
            assert hasattr(LegacyLaplacianNB(), method)

    def test_legacy_still_functional(self):
        """Test that legacy version still works for backward compatibility."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress deprecation warnings for this test

            from laplaciannb.legacy import LaplacianNB as LegacyLaplacianNB

            # Create test data in legacy format (sets)
            X_sets = np.array([{1, 2, 3}, {4, 5, 6}, {1, 4, 7}], dtype=object)
            y = np.array([0, 1, 0])

            # Should work without errors
            clf = LegacyLaplacianNB()
            clf.fit(X_sets, y)
            predictions = clf.predict(X_sets)
            probabilities = clf.predict_proba(X_sets)

            assert predictions.shape == (3,)
            assert probabilities.shape == (3, 2)  # Binary classification

    def test_new_version_functional(self):
        """Test that new version works with sklearn format."""
        from laplaciannb import LaplacianNB
        from laplaciannb.fingerprint_utils import convert_fingerprints

        # Create test data
        X_sets = [{1, 2, 3}, {4, 5, 6}, {1, 4, 7}]
        y = [0, 1, 0]

        # Convert to sklearn format
        X = convert_fingerprints(X_sets, n_bits=10)

        # Should work without errors
        clf = LaplacianNB()
        clf.fit(X, y)
        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X)

        assert predictions.shape == (3,)
        assert probabilities.shape == (3, 2)  # Binary classification


class TestMigrationSupport:
    """Test migration support features."""

    def test_explicit_legacy_import_required(self):
        """Test that legacy version requires explicit import from legacy module."""
        # Importing from main module should give new version
        from laplaciannb import LaplacianNB as MainLaplacianNB

        # Importing from legacy should give legacy version (with warning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from laplaciannb.legacy import LaplacianNB as LegacyLaplacianNB

        # They should be different
        assert MainLaplacianNB is not LegacyLaplacianNB

    def test_warning_messages_helpful(self):
        """Test that warning messages provide helpful migration information."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            from laplaciannb.legacy import LaplacianNB

            LaplacianNB()

            # Should have warnings with helpful information
            assert len(w) >= 1

            # Check warning content
            warning_messages = [str(warning.message) for warning in w]
            combined_message = " ".join(warning_messages)

            # Should mention new version
            assert "sklearn-compatible" in combined_message
            assert "from laplaciannb import LaplacianNB" in combined_message
            assert "DEPRECATED" in combined_message
            assert "REMOVED" in combined_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
