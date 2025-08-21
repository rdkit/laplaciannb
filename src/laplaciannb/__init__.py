"""LaplacianNB: Naive Bayes classifier for laplacian modified models.

This package provides both the modern sklearn-compatible implementation (recommended)
and the legacy implementation for backward compatibility.

Recommended usage:
    from laplaciannb import LaplacianNB  # Modern sklearn-compatible version

Legacy usage (deprecated):
    from laplaciannb.legacy import LaplacianNB  # Legacy version (will be removed)

The modern implementation offers:
- Full sklearn compatibility (pipelines, cross-validation, grid search)
- Memory-efficient sparse matrix support
- Better error handling and validation
- Consistent API with other sklearn estimators
- Enhanced fingerprint utility functions
"""

from .bayes import LaplacianNB
from .fingerprint_utils import rdkit_to_csr


__version__ = "0.8.0"
__all__ = [
    "LaplacianNB",
    "rdkit_to_csr",
]
