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

from .fingerprint_utils import (
    FingerprintTransformer,
    RDKitFingerprintConverter,
    convert_fingerprints,
    rdkit_sparse_to_csc,
    rdkit_sparse_to_csr,
    rdkit_sparse_to_dense,
    rdkit_sparse_to_numpy,
    rdkit_sparse_to_sklearn,
)
from .LaplacianNB import LaplacianNB


__version__ = "0.7.0"
__all__ = [
    "LaplacianNB",
    "FingerprintTransformer",
    "RDKitFingerprintConverter",
    "convert_fingerprints",
    "rdkit_sparse_to_dense",
    "rdkit_sparse_to_csr",
    "rdkit_sparse_to_csc",
    "rdkit_sparse_to_numpy",
    "rdkit_sparse_to_sklearn",
]
