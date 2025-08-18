# LaplacianNB Deprecation Timeline

## Overview

This document outlines the deprecation timeline for the legacy LaplacianNB implementation and the transition to the new sklearn-compatible version.

## Migration Strategies

### Immediate Migration (Recommended)
```python
# Before (legacy)
from laplaciannb.legacy import LaplacianNB
X_sets = [...]  # Sets of bit indices
clf = LaplacianNB()
clf.fit(X_sets, y)

# After (modern)
from laplaciannb import LaplacianNB
from laplaciannb.fingerprint_utils import convert_fingerprints
X = convert_fingerprints(X_sets, n_bits=size)
clf = LaplacianNB()
clf.fit(X, y)
```

### Gradual Migration
```python
# Phase 1: Suppress warnings while testing
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="laplaciannb.legacy")

# Phase 2: Test both implementations side by side
from laplaciannb import LaplacianNB as NewLaplacianNB
from laplaciannb.legacy import LaplacianNB as LegacyLaplacianNB

# Phase 3: Switch to new implementation
from laplaciannb import LaplacianNB
```

### Pipeline Migration
```python
# Before: Custom preprocessing
X_processed = preprocess_fingerprints(X_raw)
clf = LegacyLaplacianNB()

# After: sklearn pipeline
from sklearn.pipeline import Pipeline
from laplaciannb import LaplacianNB, FingerprintTransformer

pipeline = Pipeline([
    ('fingerprints', FingerprintTransformer(n_bits=2048)),
    ('classifier', LaplacianNB())
])
```

## Version Compatibility Matrix

| Version | Legacy Available | New Available | Default Import | Warnings |
|---------|------------------|---------------|----------------|----------|
| v0.7.0  | ✅ `legacy` module | ✅ Main module | New | Future |
| v1.0.0  | ❌ Removed | ✅ Main module | New | None |


## FAQ

### Q: How long do I have to migrate?
**A:** Legacy support will be removed in v1.0.0. We recommend migrating immediately to benefit from new features and better performance.

### Q: Are the results identical between versions?
**A:** Yes, both implementations are tested for compatibility and produce identical results.

### Q: Can I use both versions in the same project?
**A:** Yes, during the transition period. Import them with different names:
```python
from laplaciannb import LaplacianNB as NewLaplacianNB
from laplaciannb.legacy import LaplacianNB as LegacyLaplacianNB
```

### Q: What if I find bugs in the new implementation?
**A:** Please file an issue on GitHub. During the transition period, you can use the legacy version as a fallback.

### Q: Will there be breaking changes in the new implementation?
**A:** The new implementation follows sklearn conventions and semantic versioning. Breaking changes will only occur in major version releases.

## Migration Resources

1. **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - Comprehensive migration instructions
2. **[examples/](examples/)** - Example notebooks showing both versions
3. **[tests/test_compatibility.py](tests/test_compatibility.py)** - Compatibility validation
4. **GitHub Issues** - Community support and bug reports
