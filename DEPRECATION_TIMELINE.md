# LaplacianNB Deprecation Timeline

## Overview

This document outlines the deprecation timeline for the legacy LaplacianNB implementation and the transition to the new sklearn-compatible version.

## Timeline

### Phase 1: Soft Deprecation (v0.6.1 - Current)
**Status: ✅ COMPLETED**

- ✅ Legacy implementation moved to `laplaciannb.legacy` module
- ✅ New sklearn-compatible implementation available as main import
- ✅ Deprecation warnings issued on legacy import and instantiation
- ✅ Comprehensive migration guide provided
- ✅ Both versions fully functional and tested for compatibility

**User Impact:** 
- Minimal - existing code continues to work
- Deprecation warnings guide users to new implementation
- New users encouraged to use modern implementation

### Phase 2: Strong Deprecation (v0.7.0 - Planned)
**Target: Next minor release**

**Changes:**
- ⚠️ Increase severity of deprecation warnings to `FutureWarning`
- ⚠️ Add warnings to key methods (`fit`, `predict`, `predict_proba`)
- ⚠️ Update documentation to prominently feature new implementation
- ⚠️ Legacy examples moved to separate section

**Implementation:**
```python
# Additional warnings in legacy methods
def fit(self, X, y, sample_weight=None):
    warnings.warn(
        "Legacy LaplacianNB.fit() is deprecated. "
        "Use the new sklearn-compatible implementation.",
        FutureWarning,
        stacklevel=2
    )
    # ... existing implementation
```

**User Impact:**
- More visible warnings during usage
- Clear guidance on migration path
- Existing code still functional

### Phase 3: Final Warning (v0.8.0 - Planned)
**Target: Next major feature release**

**Changes:**
- 🚨 Raise `PendingDeprecationWarning` on all legacy method calls
- 🚨 Add warning counters and migration statistics
- 🚨 Legacy module requires explicit environment variable to suppress warnings
- 🚨 Documentation clearly states removal timeline

**Implementation:**
```python
import os
if not os.environ.get('LAPLACIANNB_ALLOW_LEGACY', False):
    warnings.warn(
        "Legacy LaplacianNB will be REMOVED in v1.0.0. "
        "Set LAPLACIANNB_ALLOW_LEGACY=1 to suppress this warning.",
        PendingDeprecationWarning,
        stacklevel=2
    )
```

**User Impact:**
- Strong pressure to migrate
- Clear timeline for removal
- Option to suppress warnings for gradual migration

### Phase 4: Removal (v1.0.0 - Planned)
**Target: Major version release**

**Changes:**
- 🗑️ Complete removal of `laplaciannb.legacy` module
- 🗑️ Legacy implementation no longer available
- 🗑️ Clean codebase with only modern implementation
- 📚 Migration guide archived as historical reference

**User Impact:**
- Breaking change for users still on legacy
- Clean, modern codebase
- Better performance and maintainability

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

## Support Policy

### Current Support (v0.6.x - v0.9.x)
- **Legacy:** Bug fixes only, no new features
- **New:** Full feature development and support
- **Migration:** Comprehensive documentation and examples

### Future Support (v1.0.0+)
- **Legacy:** Not available
- **New:** Full support and active development
- **Migration:** Historical documentation only

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

## Communication Plan

### Documentation Updates
- README.md updated with migration information
- API documentation shows new implementation first
- Legacy documentation clearly marked as deprecated

### Community Outreach
- Release notes highlight deprecation timeline
- Migration examples in all documentation
- Community forums and issue tracking for migration support

### Monitoring
- Track usage patterns to understand migration progress
- Monitor GitHub issues for migration-related problems
- Collect feedback on migration experience

This timeline ensures a smooth transition while providing ample time and resources for users to migrate to the modern implementation.
