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

## ✨ Features

### 🔬 Core Algorithm
- **Laplacian-modified Naive Bayes** with enhanced smoothing for sparse data
- **Optimized for binary/boolean features** using bit index representation
- **Fast prediction** leveraging only positive bit indices
- **Robust handling** of unseen features and classes

### 🚀 Performance & Scalability
- **Memory-efficient sparse matrix support** for massive feature spaces (2^32 features)
- **Lossless RDKit fingerprint conversion** with bit reinterpretation
- **Automatic sparsity detection** and optimization
- **Parallel processing** compatible with joblib

### 🔧 sklearn Integration
- **Full sklearn ecosystem compatibility** (pipelines, cross-validation, grid search)
- **Drop-in replacement** for other Naive Bayes classifiers
- **Consistent API** with sklearn estimators
- **Custom transformers** for molecular data preprocessing

### 🧪 Molecular Informatics
- **Direct RDKit integration** for SMILES conversion
- **Morgan fingerprint support** with configurable radius
- **Chemical space analysis** capabilities
- **QSAR/SAR modeling** optimized workflows

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
For the latest development version with examples:

```sh
git clone https://github.com/rdkit/laplaciannb.git
cd laplaciannb
pip install -e ".[dev]"  # Includes development dependencies
```

### Optional Dependencies
For molecular fingerprint functionality:
```sh
pip install rdkit  # For molecular fingerprint conversion
```

For full development environment:
```sh
pip install laplaciannb[dev]  # Includes testing, linting, and examples
```

## Quick Start

### 🚀 Try the Interactive Example

Run the comprehensive quickstart example to see all features in action:

```sh
cd examples
python quickstart_example.py
```

This script demonstrates:
- RDKit molecular fingerprint conversion
- Sparse matrix handling for memory efficiency
- scikit-learn ecosystem integration
- Performance comparisons with other classifiers
- Memory efficiency demonstrations

### Recommended Usage (Modern sklearn-compatible API)

**For molecular data with RDKit:**

```python
from laplaciannb import LaplacianNB
from laplaciannb.fingerprint_utils import rdkit_to_csr

# Sample molecular data (SMILES strings)
smiles = [
    "CCO",                              # Ethanol
    "CC(=O)OC1=CC=CC=C1C(=O)O",        # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"    # Ibuprofen
]
y = [0, 1, 1]  # Activity labels

# Convert to sparse CSR matrix (memory efficient)
X = rdkit_to_csr(smiles, radius=2)
print(f"Matrix shape: {X.shape}")  # (3, 4294967296)
print(f"Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.6f}")

# Train classifier
clf = LaplacianNB(alpha=1.0)
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)
probabilities = clf.predict_proba(X)
```

**For general binary/boolean data:**

```python
import numpy as np
from scipy.sparse import csr_matrix
from laplaciannb import LaplacianNB

# Create sparse binary matrix directly
row = [0, 0, 1, 1, 2, 2]
col = [1, 5, 2, 6, 1, 3]
data = [1, 1, 1, 1, 1, 1]
X = csr_matrix((data, (row, col)), shape=(3, 10), dtype=np.bool_)
y = [0, 1, 0]

# Train and predict
clf = LaplacianNB(alpha=1.0)
clf.fit(X, y)
predictions = clf.predict(X)
probabilities = clf.predict_proba(X)
```

### sklearn Ecosystem Integration

**Full Pipeline Example:**

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from laplaciannb import LaplacianNB
from laplaciannb.fingerprint_utils import rdkit_to_csr

# Custom transformer for pipelines
class RDKitFingerprintTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, radius=2):
        self.radius = radius

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return rdkit_to_csr(X, radius=self.radius)

# Create pipeline
pipeline = Pipeline([
    ('fingerprints', RDKitFingerprintTransformer(radius=2)),
    ('classifier', LaplacianNB(alpha=1.0))
])

# Grid search
param_grid = {
    'classifier__alpha': [0.1, 1.0, 10.0],
    'fingerprints__radius': [1, 2, 3]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(smiles_data, y)  # Use SMILES directly in pipeline

# Cross-validation
cv_scores = cross_val_score(pipeline, smiles_data, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Direct sparse matrix usage (for pre-converted data)
X_sparse = rdkit_to_csr(smiles_data, radius=2)
clf = LaplacianNB(alpha=1.0)
scores = cross_val_score(clf, X_sparse, y, cv=5)
```

## 🔥 Key Features & Advantages

### Memory Efficiency
- **Sparse matrix support**: Handle 2^32 feature spaces with minimal memory
- **Lossless fingerprint conversion**: Convert RDKit fingerprints without data loss
- **Automatic sparsity detection**: Works seamlessly with both sparse and dense data

```python
# Handle massive feature spaces efficiently
X = rdkit_to_csr(smiles_list, radius=2)  # Shape: (n_samples, 4294967296)
print(f"Memory usage: {X.data.nbytes / 1024**2:.1f} MB")  # Only a few MB!
```

### Performance
- **Optimized for binary data**: Fast prediction using only positive bit indices
- **sklearn compatible**: Drop-in replacement for other Naive Bayes classifiers
- **Parallel processing**: Supports joblib parallelization

### Molecular Informatics
- **RDKit integration**: Direct conversion from molecular structures
- **Flexible fingerprints**: Support for Morgan, MACCS, and custom fingerprints
- **Chemical space analysis**: Ideal for QSAR/SAR modeling

## 📚 Examples & Tutorials

### Interactive Examples
Explore the comprehensive examples in the `/examples` directory:

- **`quickstart_example.py`**: Complete demonstration with molecular data
- **`basic_usage_tutorial.ipynb`**: Step-by-step Jupyter notebook
- **`sklearn_integration_tutorial.ipynb`**: Advanced sklearn integration
- **`bayes_tutorial.ipynb`**: Deep dive into Naive Bayes concepts

### Run the Quickstart
```sh
# Clone the repository
git clone https://github.com/rdkit/laplaciannb.git
cd laplaciannb

# Install with examples
pip install -e ".[dev]"

# Run comprehensive example
python examples/quickstart_example.py
```

### Example Outputs
The quickstart example demonstrates:
```
BASIC LAPLACIANNB USAGE
Matrix shape: (10, 4294967296)
Matrix sparsity: 0.999998
Training completed in 0.002 seconds
Test Accuracy: 1.000

SPARSE MATRIX EFFICIENCY
Radius   Features     Sparsity   Train Time   Accuracy
1        4,294,967,296 0.999999   0.001       1.000
2        4,294,967,296 0.999998   0.002       1.000
3        4,294,967,296 0.999997   0.003       1.000

MEMORY EFFICIENCY
Sparse matrix memory: 0.12 MB
Dense equivalent would require 40,000+ MB!
✓ Designed specifically for extremely sparse binary features
```
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
