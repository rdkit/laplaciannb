<h1 align="center">LaplacianNB</h1>

<p align="center">
  <b>Laplacian-modified Naive Bayes classifier for models</b><br>
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

**LaplacianNB** is a Python module developed at **Novartis AG** for a Laplacian-modified Naive Bayes classifier models, based on the scikit-learn Naive Bayes implementation.

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
- **Progress tracking** with tqdm integration for large datasets
- **Comprehensive benchmarking** tools for performance analysis
- **Large-scale processing** validated up to 100,000+ molecules
- **Reverse mapping** capabilities for feature interpretation

### 🔧 sklearn Integration
- **Drop-in replacement** for other Naive Bayes classifiers
- **Consistent API** with sklearn estimators

### 🧪 Molecular Informatics
- **Direct RDKit integration** for SMILES conversion
- **QSAR/SAR modeling** optimized workflows
- **Large-scale molecular processing** with progress tracking
- **Feature interpretation** through reverse index mapping

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
pip install -e ".[test]"  # Includes development dependencies
```

### Optional Dependencies
For molecular fingerprint functionality:
```sh
pip install rdkit  # For molecular fingerprint conversion
pip install tqdm   # For progress bars (optional but recommended)
```

For full development environment:
```sh
pip install laplaciannb[test]  # Includes testing, linting, and examples
```

## Quick Start

### 🚀 Try the Interactive Examples

Run the comprehensive examples to see all features in action:

```sh
cd examples
python simple_example.py          # Basic usage with reverse mapping
python benchmark_fingerprints.py  # Performance benchmarking
python benchmark_large_scale.py   # Large-scale testing (100K molecules)
```

These scripts demonstrate:
- RDKit molecular fingerprint conversion with progress tracking (tqdm)
- Sparse matrix handling for memory efficiency
- Performance benchmarking and scalability analysis (up to 100K molecules)
- Reverse mapping from sparse matrices to RDKit indices
- Large-scale molecular processing capabilities
- Memory efficiency analysis and sparsity reporting

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

# Convert to sparse CSR matrix (memory efficient, with progress tracking)
X = rdkit_to_csr(smiles, radius=2, show_progress=True)
print(f"Matrix shape: {X.shape}")  # (3, 4294967296)
print(f"Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.6f}")

# Train classifier
clf = LaplacianNB(alpha=1.0)
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)
probabilities = clf.predict_proba(X)

# Optional: Reverse mapping for feature interpretation
def uint32_to_rdkit_index(uint32_index):
    """Convert sparse matrix index back to original RDKit fingerprint bit."""
    if uint32_index >= 2**31:
        return int(uint32_index) - 2**32  # Convert back to signed int32
    else:
        return int(uint32_index)

# Example: Get active features for first molecule
mol_idx = 0
start_idx, end_idx = X.indptr[mol_idx], X.indptr[mol_idx + 1]
sparse_indices = X.indices[start_idx:end_idx]
rdkit_indices = [uint32_to_rdkit_index(idx) for idx in sparse_indices]
print(f"RDKit fingerprint indices: {rdkit_indices[:10]}...")  # Show first 10
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

### Performance Benchmarking

**Built-in benchmarking tools for performance analysis:**

```python
from laplaciannb.fingerprint_utils import benchmark_fingerprint_conversion, benchmark_large_scale_conversion

# Standard benchmarking with different parameters
benchmark_fingerprint_conversion(
    n_molecules=10000,
    radii=[1, 2, 3],
    molecules_per_test=[1000, 5000, 10000]
)

# Large-scale performance validation
results = benchmark_large_scale_conversion(
    target_molecules=100000,
    test_sizes=[1000, 10000, 50000, 100000],
    radius=2,
    sample_diversity=True
)

# Results show linear scaling and high throughput
print(f"Peak performance: {max(r['rate'] for r in results):,.0f} molecules/second")
print(f"Memory efficiency: >99.999% sparsity maintained")
```

### Feature Interpretation & Reverse Mapping

**Trace predictions back to molecular substructures:**

```python
from rdkit.Chem import rdFingerprintGenerator

def uint32_to_rdkit_index(uint32_index):
    """Convert sparse matrix index back to RDKit fingerprint bit."""
    if uint32_index >= 2**31:
        return int(uint32_index) - 2**32
    return int(uint32_index)

# Train model and get predictions
clf.fit(X, y)
predictions = clf.predict(X)

# For each molecule, show which fingerprint bits influenced prediction
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
for i, smiles_str in enumerate(smiles):
    # Get active features from sparse matrix
    start_idx, end_idx = X.indptr[i], X.indptr[i + 1]
    sparse_indices = X.indices[start_idx:end_idx]
    rdkit_indices = [uint32_to_rdkit_index(idx) for idx in sparse_indices]

    # Compare with original RDKit fingerprint
    mol = Chem.MolFromSmiles(smiles_str)
    original_fp = mfpgen.GetSparseFingerprint(mol)
    original_indices = sorted(original_fp.GetOnBits())

    print(f"Molecule: {smiles_str}")
    print(f"Prediction: {predictions[i]}")
    print(f"Active features: {len(rdkit_indices)} bits")
    print(f"Round-trip validation: {sorted(rdkit_indices) == original_indices}")
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
print(f"Sparsity: {1 - X.nnz / X.size:.6f}")  # >99.999% sparse
```

### Performance & Benchmarking
- **Optimized for binary data**: Fast prediction using only positive bit indices
- **sklearn compatible**: Drop-in replacement for other Naive Bayes classifiers
- **Built-in benchmarking**: Comprehensive performance analysis tools
- **Scalability tested**: Validated with datasets up to 100,000+ molecules

```python
from laplaciannb.fingerprint_utils import benchmark_fingerprint_conversion

# Benchmark conversion performance
benchmark_fingerprint_conversion(
    n_molecules=10000,
    radii=[1, 2, 3],
    molecules_per_test=[1000, 5000, 10000]
)
```

### Feature Interpretation
- **Reverse mapping**: Convert sparse matrix indices back to RDKit fingerprint bits
- **Chemical insights**: Identify which molecular features drive predictions
- **Debugging support**: Trace predictions back to original molecular substructures

```python
# Map sparse matrix features back to RDKit fingerprint indices
def uint32_to_rdkit_index(uint32_index):
    if uint32_index >= 2**31:
        return int(uint32_index) - 2**32
    return int(uint32_index)

# Get active features for interpretation
active_features = [uint32_to_rdkit_index(idx) for idx in sparse_indices]
```

## 📚 Examples & Tutorials

### Interactive Examples
Explore the comprehensive examples in the `/examples` directory:

- **`simple_example.py`**: Complete demonstration with reverse mapping functionality
- **`benchmark_fingerprints.py`**: Performance benchmarking with different parameters
- **`benchmark_large_scale.py`**: Large-scale testing up to 100,000 molecules
- **Jupyter notebooks**: Step-by-step tutorials for advanced usage

### Run the Examples
```sh
# Clone the repository
git clone https://github.com/rdkit/laplaciannb.git
cd laplaciannb

# Install with examples
pip install -e ".[dev]"

# Run basic example with reverse mapping
python examples/simple_example.py

# Test performance benchmarking
python examples/benchmark_fingerprints.py

# Large-scale performance validation
python examples/benchmark_large_scale.py
```

### Example Outputs
The benchmark examples demonstrate impressive performance:
```
FINGERPRINT CONVERSION BENCHMARK
===============================================
Converting 100,000 molecules to Morgan fingerprints...
100%|██████████| 100000/100000 [00:03<00:00, 31567.21molecules/s]

Molecules    Time (s)   Rate (mol/s)   Memory (MB)   Sparsity
------------ ---------- -------------- ------------- -----------
1,000        0.032      31,250         0.61          0.999998
10,000       0.318      31,447         6.12          0.999998
100,000      3.167      31,567         61.23         0.999998

MEMORY EFFICIENCY ANALYSIS
✓ Sparse matrix memory: 61.23 MB
✓ Dense equivalent would require: 1,600,000+ MB
✓ Memory savings: 99.999997%
✓ Linear scaling confirmed up to 100K molecules
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
│   ├── bayes.py               # Modern sklearn-compatible implementation
│   ├── fingerprint_utils.py   # Enhanced conversion utilities with benchmarking
│   └── legacy/                # Deprecated legacy API
├── examples/                  # Comprehensive examples
│   ├── simple_example.py      # Basic usage with reverse mapping
│   ├── benchmark_fingerprints.py    # Performance benchmarking
│   └── benchmark_large_scale.py     # Large-scale testing (100K molecules)
├── tests/                     # Comprehensive test suite
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

### v0.8.0 (Latest)
- **Enhanced fingerprint conversion** with tqdm progress tracking
- **Comprehensive benchmarking tools** for performance analysis
- **Large-scale processing support** validated up to 100,000+ molecules
- **Reverse mapping functionality** for feature interpretation
- **Improved examples** with practical demonstrations
- **Better memory efficiency** reporting and sparsity analysis
- **Code quality improvements** with ruff formatting and pre-commit hooks

### v0.7.0
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
