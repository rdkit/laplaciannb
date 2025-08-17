<h1 align="center">LaplacianNB</h1>

<p align="center">
  <b>Naive Bayes classifier for Laplacian-modified models</b><br>
  <i>Efficient, scikit-learn compatible, and designed for binary/boolean data</i>
</p>

<p align="center">
  <a href="https://pypi.org/project/laplaciannb/"><img src="https://img.shields.io/pypi/v/laplaciannb.svg?logo=pypi&label=PyPI&logoColor=gold" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/laplaciannb/"><img src="https://img.shields.io/pypi/dm/laplaciannb.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold" alt="PyPI Downloads"></a>
  <a href="https://pypi.org/project/laplaciannb/"><img src="https://img.shields.io/pypi/pyversions/laplaciannb.svg?logo=python&label=Python&logoColor=gold" alt="Python Versions"></a>
</p>

**LaplacianNB** is a Python module developed at **Novartis AG** for a Naive Bayes classifier for Laplacian-modified models, based on the scikit-learn Naive Bayes implementation.

This classifier is ideal for binary/boolean data, using only the indices of positive bits for efficient prediction. The algorithm was first implemented in Pipeline Pilot and KNIME.

The package also includes comprehensive utilities for converting RDKit molecular fingerprints to sklearn-compatible formats.

---

## 🚀 Features

- **Naive Bayes classifier** for Laplacian-modified models
- **Optimized for binary/boolean data**
- **Fast prediction** using indices of positive bits
- **scikit-learn compatible API**
- **RDKit fingerprint conversion utilities**
- **Support for sparse and dense data formats**
- Lightweight and easy to integrate

---

## 📦 Installation

Install the latest release from PyPI:

```sh
pip install laplaciannb
```

## 🔬 Quick Start

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

## 📚 Literature

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

## 👤 Authors & Maintainers

- **Bartosz Baranowski** (bartosz.baranowski@novartis.com)  
- **Edgar Harutyunyan** (edgar.harutyunyan_ext@novartis.com)

---

## 📝 Changelog

- `v0.6.1` - Fixes for scikit-learn 1.7, rdkit 2025+ compatibility, move to uv build
- `v0.6.0` - Move to pdm build
- `v0.5.0` - Initial release

---

## 📄 License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.
