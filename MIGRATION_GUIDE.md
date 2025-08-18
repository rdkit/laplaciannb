# LaplacianNB Migration Guide

## Overview

LaplacianNB has been modernized with a new sklearn-compatible implementation. This guide helps you migrate from the legacy version to the new recommended version.

## Quick Migration

### Old Way (Deprecated)
```python
from laplaciannb.legacy import LaplacianNB  # ⚠️ DEPRECATED
```

### New Way (Recommended)
```python
from laplaciannb import LaplacianNB  # ✅ RECOMMENDED
```

## Key Differences

### Input Data Format

**Legacy Implementation:**
- Expects fingerprints as sets, lists, or dictionaries
- Custom input validation
- Limited to specific data formats

```python
# Legacy - fingerprints as sets
X_sets = [
    {1, 5, 10, 15},
    {2, 6, 11, 16},
    {1, 3, 7, 12}
]
```

**New Implementation:**
- Accepts standard sklearn input formats (sparse/dense matrices)
- Full sklearn input validation
- Seamless integration with sklearn ecosystem

```python
# New - sklearn-compatible sparse/dense matrices
from laplaciannb.fingerprint_utils import convert_fingerprints

X_sklearn = convert_fingerprints(X_sets, n_bits=2048, output_format='csr')
# or use FingerprintTransformer in pipelines
```

### API Changes

**Legacy:**
```python
from laplaciannb.legacy import LaplacianNB

clf = LaplacianNB(alpha=1.0)
clf.fit(X_sets, y)
predictions = clf.predict(X_sets)
```

**New:**
```python
from laplaciannb import LaplacianNB
from laplaciannb.fingerprint_utils import convert_fingerprints

# Convert fingerprints to sklearn format
X = convert_fingerprints(X_sets, n_bits=2048)

clf = LaplacianNB(alpha=1.0)
clf.fit(X, y)
predictions = clf.predict(X)
```

### Enhanced Features in New Version

1. **sklearn Ecosystem Integration:**
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.model_selection import GridSearchCV
   from laplaciannb import LaplacianNB, FingerprintTransformer
   
   # Pipeline support
   pipeline = Pipeline([
       ('fingerprints', FingerprintTransformer(n_bits=2048)),
       ('classifier', LaplacianNB())
   ])
   
   # Grid search support
   param_grid = {'classifier__alpha': [0.1, 1.0, 10.0]}
   grid_search = GridSearchCV(pipeline, param_grid, cv=5)
   ```

2. **Memory-Efficient Sparse Matrices:**
   ```python
   # Automatic sparse matrix handling for large fingerprints
   X_sparse = convert_fingerprints(fingerprints, n_bits=16384, output_format='csr')
   clf = LaplacianNB()
   clf.fit(X_sparse, y)  # Memory efficient for sparse data
   ```

3. **Better Error Handling:**
   ```python
   # Comprehensive input validation
   # Clear error messages
   # Proper sklearn-style exceptions
   ```

## Migration Steps

### Step 1: Update Imports
```python
# Before
from laplaciannb.legacy import LaplacianNB

# After  
from laplaciannb import LaplacianNB
from laplaciannb.fingerprint_utils import convert_fingerprints
```

### Step 2: Convert Input Data
```python
# Before - fingerprints as sets/lists
X_fingerprints = [...]  # Your fingerprint data

# After - convert to sklearn format
X = convert_fingerprints(X_fingerprints, n_bits=your_fingerprint_size)
```

### Step 3: Update Model Usage
```python
# Both versions use the same basic API
clf = LaplacianNB(alpha=1.0)
clf.fit(X, y)
predictions = clf.predict(X)
probabilities = clf.predict_proba(X)
```

### Step 4: Leverage New Features (Optional)
```python
# Use in sklearn pipelines
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('classifier', LaplacianNB())
])

# Cross-validation
scores = cross_val_score(pipeline, X, y, cv=5)
```

## Common Migration Issues

### Issue 1: Input Format Mismatch
**Problem:** Getting errors about input format

**Solution:** Use fingerprint utilities to convert data
```python
from laplaciannb.fingerprint_utils import convert_fingerprints

# Convert sets to sklearn format
X_sklearn = convert_fingerprints(your_fingerprint_sets, n_bits=2048)
```

### Issue 2: Memory Issues with Large Fingerprints
**Problem:** Running out of memory with large dense matrices

**Solution:** Use sparse matrices (default behavior)
```python
# Default output is memory-efficient sparse CSR matrix
X_sparse = convert_fingerprints(fingerprints, n_bits=16384)  # Uses CSR by default
```

### Issue 3: Different Prediction Results
**Problem:** Getting slightly different results

**Solution:** This should not happen - both implementations are tested for compatibility. If you encounter this, please file an issue.

## Compatibility Guarantees

- **Identical Results:** New implementation produces identical predictions to legacy version
- **Backward Compatibility:** Legacy version remains available in `laplaciannb.legacy`
- **Migration Period:** Legacy version will be maintained until sufficient adoption of new version

## Testing Your Migration

Use our compatibility test to verify your migration:

```python
import numpy as np
from laplaciannb import LaplacianNB as LaplacianNB_New
from laplaciannb.legacy import LaplacianNB as LaplacianNB_Legacy
from laplaciannb.fingerprint_utils import convert_fingerprints

# Your test data
X_sets = [...]  # Your fingerprint sets
y = [...]       # Your labels

# Test both implementations
clf_legacy = LaplacianNB_Legacy(alpha=1.0)
clf_legacy.fit(np.array(X_sets, dtype=object), y)
pred_legacy = clf_legacy.predict(np.array(X_sets, dtype=object))

X_sklearn = convert_fingerprints(X_sets, n_bits=your_n_bits)
clf_new = LaplacianNB_New(alpha=1.0)
clf_new.fit(X_sklearn, y)
pred_new = clf_new.predict(X_sklearn)

# Verify identical results
assert np.array_equal(pred_legacy, pred_new), "Predictions should be identical"
print("✓ Migration successful - identical predictions!")
```

## Getting Help

- **Documentation:** See example notebooks in `examples/` directory
- **Issues:** File issues on GitHub if you encounter migration problems
- **Examples:** Check `examples/sklearn_integration_tutorial.ipynb` for sklearn usage patterns

## Timeline

- **v0.7.0:** Increase deprecation warning severity
- **v1.0.0:** Legacy version removal (planned)

The migration is designed to be straightforward while providing significant benefits in terms of sklearn ecosystem integration and performance.
