"""
Legacy LaplacianNB implementation.

DEPRECATED: This module contains the legacy LaplacianNB implementation.
Please use the new sklearn-compatible version instead:

    from laplaciannb import LaplacianNB  # New version (recommended)

instead of:

    from laplaciannb.legacy import LaplacianNB  # Old version (deprecated)

The new implementation offers:
- Full sklearn compatibility (pipelines, cross-validation, grid search)
- Memory-efficient sparse matrix support
- Better error handling and validation
- Consistent API with other sklearn estimators
- Enhanced fingerprint utility functions

The legacy version will be removed in a future release.
"""

import warnings

from .LaplacianNB import LaplacianNB


# Issue strong deprecation warning when legacy module is imported
warnings.warn(
    "\n" + "=" * 80 + "\n"
    "DEPRECATION WARNING: Legacy LaplacianNB Implementation\n" + "=" * 80 + "\n"
    "You are importing from the DEPRECATED legacy LaplacianNB module.\n"
    "This implementation will be REMOVED in a future release.\n\n"
    "PLEASE MIGRATE to the new sklearn-compatible version:\n\n"
    "  ✅ RECOMMENDED:\n"
    "    from laplaciannb import LaplacianNB\n"
    "    from laplaciannb.fingerprint_utils import convert_fingerprints\n\n"
    "  ❌ DEPRECATED (current usage):\n"
    "    from laplaciannb.legacy import LaplacianNB\n\n"
    "The new implementation provides:\n"
    "• Full sklearn ecosystem compatibility\n"
    "• Memory-efficient sparse matrix support\n"
    "• Better performance and error handling\n"
    "• Enhanced fingerprint conversion utilities\n\n"
    "See MIGRATION_GUIDE.md for detailed migration instructions.\n" + "=" * 80,
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LaplacianNB"]
