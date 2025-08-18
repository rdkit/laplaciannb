<h1 align="center">LaplacianNB</h1>

<p align="center">
  <b>Naive Bayes classifier for Laplacian-modified models</b><br>
  <i>Efficient, scikit-learn compatible, and designed for binary/boolean data</i>
</p>

<p align="center">
  <a href="https://pypi.org/project/laplaciannb/"><img src="https://img.shields.io/pypi/v/laplaciannb.svg" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/laplaciannb/"><img src="https://img.shields.io/pypi/dm/laplaciannb.svg" alt="PyPI Downloads"></a>
  <a href="https://img.shields.io/pypi/pyversions/laplaciannb"><img src="https://img.shields.io/pypi/pyversions/laplaciannb.svg" alt="Python Versions"></a>
  <a href="https://github.com/rdkit/laplaciannb/actions/workflows/ruff.yml"><img src="https://github.com/rdkit/laplaciannb/workflows/Code%20Quality%20Checks/badge.svg" alt="Code Quality"></a>
  <a href="https://github.com/rdkit/laplaciannb/actions/workflows/coverage.yml"><img src="https://github.com/rdkit/laplaciannb/workflows/Test%20Coverage/badge.svg" alt="Test Coverage"></a>
  <a href="https://github.com/rdkit/laplaciannb/actions/workflows/security.yml"><img src="https://github.com/rdkit/laplaciannb/workflows/Security%20Scanning/badge.svg" alt="Security Scan"></a>
  <a href="https://github.com/rdkit/laplaciannb/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg" alt="License"></a>
  <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen" alt="pre-commit"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

**LaplacianNB** is a Python module developed at **Novartis AG** for a Naive Bayes classifier for Laplacian-modified models, based on the scikit-learn Naive Bayes implementation.

This classifier is ideal for binary/boolean data, using only the indices of positive bits for efficient prediction. The algorithm was first implemented in Pipeline Pilot and KNIME.

The package includes both a **modern sklearn-compatible implementation** (recommended) and a legacy version for backward compatibility.

---

## Features

- **Modern sklearn-compatible implementation** with full ecosystem integration
- **Optimized for binary/boolean data** with fast prediction using indices of positive bits
- **RDKit fingerprint conversion utilities** for molecular data
- **Support for sparse and dense data formats**
- **Memory-efficient sparse matrix handling**
- Lightweight and easy to integrate

---

## Installation

### Stable Release
Install the latest stable release from PyPI:

```sh
pip install laplaciannb
```

### Development Version
Get the latest features with development releases:

```sh
pip install --pre laplaciannb
```

### From Source
For the latest development version:

```sh
git clone https://github.com/rdkit/laplaciannb.git
cd laplaciannb
pip install -e .
```

## Quick Start

### Recommended Usage (Modern sklearn-compatible API)

```python
import numpy as np
from laplaciannb import LaplacianNB
from laplaciannb.fingerprint_utils import convert_fingerprints

# Convert fingerprint data to sklearn format
fingerprints = [
    {1, 5, 10, 15},      # Fingerprint as set of bit indices
    {2, 6, 11, 16},      # Each set represents active bits
    {1, 3, 7, 12}
]
X = convert_fingerprints(fingerprints, n_bits=20)
y = [0, 1, 0]

# Train and predict
clf = LaplacianNB(alpha=1.0)
clf.fit(X, y)
predictions = clf.predict(X)
probabilities = clf.predict_proba(X)
```

### sklearn Ecosystem Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from laplaciannb import LaplacianNB, FingerprintTransformer

# Create pipeline
pipeline = Pipeline([
    ('fingerprints', FingerprintTransformer(n_bits=2048)),
    ('classifier', LaplacianNB())
])

# Grid search
param_grid = {
    'classifier__alpha': [0.1, 1.0, 10.0],
    'fingerprints__output_format': ['csr', 'dense']
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(fingerprints, y)

# Cross-validation
cv_scores = cross_val_score(pipeline, fingerprints, y, cv=5)
```

### Legacy Usage (Deprecated)

> **⚠️ DEPRECATION NOTICE:** The legacy API is deprecated and will be removed in a future release. Please migrate to the modern sklearn-compatible API above.

```python
# For backward compatibility only - will show deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from laplaciannb.legacy import LaplacianNB as LegacyLaplacianNB

# Legacy format (sets of bit indices)
X_sets = np.array([{1, 5, 10}, {2, 6, 11}, {1, 3, 7}], dtype=object)
y = [0, 1, 0]

clf = LegacyLaplacianNB(alpha=1.0)
clf.fit(X_sets, y)
predictions = clf.predict(X_sets)
```

---

## Migration Guide

**Migrating from legacy to modern implementation is easy:**

1. **Update imports:**
   ```python
   # Before (deprecated)
   from laplaciannb.legacy import LaplacianNB

   # After (recommended)
   from laplaciannb import LaplacianNB
   from laplaciannb.fingerprint_utils import convert_fingerprints
   ```

2. **Convert input data:**
   ```python
   # Convert fingerprint sets to sklearn format
   X = convert_fingerprints(your_fingerprint_sets, n_bits=your_size)
   ```

3. **Same API for basic usage:**
   ```python
   clf = LaplacianNB(alpha=1.0)
   clf.fit(X, y)
   predictions = clf.predict(X)
   ```

📖 **Detailed migration instructions:** [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
📅 **Deprecation timeline:** [DEPRECATION_TIMELINE.md](DEPRECATION_TIMELINE.md)

---

### Basic Usage with LaplacianNB

```python
import numpy as np
from laplaciannb import LaplacianNB

# Create sample data (sets of positive bit indices)
X = np.array([
    {1, 5, 10, 15},      # Sample 1: bits 1,5,10,15 are on
    {2, 6, 11, 16},      # Sample 2: bits 2,6,11,16 are on
    {1, 3, 7, 12},       # Sample 3: bits 1,3,7,12 are on
], dtype=object)
y = np.array([0, 1, 0])  # Class labels

# Train the classifier
clf = LaplacianNB()
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)
probabilities = clf.predict_proba(X)
```

### RDKit Fingerprint Integration

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from laplaciannb import LaplacianNB, convert_fingerprints

# Generate molecular fingerprints
molecules = [Chem.MolFromSmiles(smi) for smi in ['CCO', 'CC', 'CCC']]
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in molecules]

# Convert to sklearn-compatible format
X = convert_fingerprints(fingerprints, output_format='csr')
y = [0, 1, 0]

# Train classifier
clf = LaplacianNB()
clf.fit(X, y)
```

### Advanced Fingerprint Conversion

```python
from laplaciannb import RDKitFingerprintConverter

# Create converter with custom settings
converter = RDKitFingerprintConverter(
    n_bits=2048,
    output_format='auto',  # Automatically choose sparse/dense
    dtype=np.float32
)

# Convert fingerprints
X_dense = converter.to_dense(fingerprints)
X_sparse = converter.to_csr(fingerprints)

# Get statistics
stats = converter.get_statistics(fingerprints)
print(f"Sparsity: {stats['sparsity']:.2%}")
print(f"Average on-bits: {stats['avg_on_bits']:.1f}")
```

---

## Development

### Contributing

We welcome contributions! Please see our development setup:

```bash
# Clone the repository
git clone https://github.com/rdkit/laplaciannb.git
cd laplaciannb

# Install in development mode with test dependencies
pip install -e .[test]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run quality checks
pre-commit run --all-files
```

### CI/CD Pipeline

- **Code Quality:** Ruff linting and formatting
- **Testing:** Multi-Python version testing with coverage
- **Security:** Bandit security scanning
- **Auto-publishing:** Development versions on merge to develop
- **Dependency Management:** Dependabot for automated updates

### Project Structure

```
laplaciannb/
├── src/laplaciannb/           # Main package
│   ├── LaplacianNB_new.py     # Modern implementation
│   ├── fingerprint_utils.py   # Conversion utilities
│   └── legacy/                # Deprecated legacy API
├── tests/                     # Test suite
├── .github/                   # CI/CD workflows
└── docs/                      # Documentation
```

---

## Literature

```
Nidhi; Glick, M.; Davies, J. W.; Jenkins, J. L. Prediction of biological targets
for compounds using multiple-category Bayesian models trained on chemogenomics
databases. J. Chem. Inf. Model. 2006, 46, 1124– 1133,
https://doi.org/10.1021/ci060003g

Lam PY, Kutchukian P, Anand R, et al. Cyp1 inhibition prevents doxorubicin-induced cardiomyopathy
in a zebrafish heart-failure model. Chem Bio Chem. 2020:cbic.201900741.
https://doi.org/10.1002/cbic.201900741
```

---

## Authors & Maintainers

- **Bartosz Baranowski** (bartosz.baranowski@novartis.com)
- **Edgar Harutyunyan** (edgar.harutyunyan_ext@novartis.com)

---

## Changelog

### v0.7.0 (Latest)
- **Sklearn integration** handling standard sklearn input allowing for full integration with sklearn framework
- **Enhanced deprecation strategy** with comprehensive migration support
- **Legacy input detection** in new version with helpful error messages
- **Dependabot configuration** for automated dependency updates

### v0.6.1
- Fixes for scikit-learn 1.7, rdkit 2025+ compatibility
- Move to uv build system

### v0.6.0
- Move to pdm build system

### v0.5.0
- Initial public release

---

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
