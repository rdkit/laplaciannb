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

---

## 🚀 Features

- **Naive Bayes classifier** for Laplacian-modified models
- **Optimized for binary/boolean data**
- **Fast prediction** using indices of positive bits
- **scikit-learn compatible API**
- Lightweight and easy to integrate

---

## 📦 Installation

Install the latest release from PyPI:

```sh
pip install laplaciannb
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
