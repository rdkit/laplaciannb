from typing import Any, Dict, Optional, Union

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def rdkit_sparse_to_dense(fingerprint, n_bits: int = 2048, dtype=np.float32) -> np.ndarray:
    """Convert a single RDKit sparse fingerprint to dense numpy array.

    Parameters
    ----------
    fingerprint : various RDKit fingerprint types
        Can be:
        - RDKit ExplicitBitVect
        - RDKit SparseBitVect
        - RDKit IntSparseIntVect
        - UIntSparseIntVect
        - LongSparseIntVect
        - Set of on-bit indices
        - Dict mapping bit indices to counts
        - List/tuple of on-bit indices

    n_bits : int, default=2048
        Size of the output fingerprint vector.

    dtype : numpy dtype, default=np.float32
        Data type of the output array.

    Returns
    -------
    np.ndarray
        Dense numpy array of shape (n_bits,) with binary or count values.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    >>> dense_fp = rdkit_sparse_to_dense(fp, n_bits=2048)
    """
    dense = np.zeros(n_bits, dtype=dtype)

    if fingerprint is None:
        return dense

    # Handle RDKit BitVect types
    if hasattr(fingerprint, "GetOnBits"):
        # ExplicitBitVect or SparseBitVect
        for bit_idx in fingerprint.GetOnBits():
            if 0 <= bit_idx < n_bits:
                dense[bit_idx] = 1.0

    # Handle RDKit SparseIntVect types
    elif hasattr(fingerprint, "GetNonzeroElements"):
        # IntSparseIntVect, UIntSparseIntVect, LongSparseIntVect
        for bit_idx, count in fingerprint.GetNonzeroElements().items():
            if 0 <= bit_idx < n_bits:
                dense[bit_idx] = float(count)

    # Handle Python set (set of on-bits)
    elif isinstance(fingerprint, set):
        for bit_idx in fingerprint:
            if 0 <= bit_idx < n_bits:
                dense[bit_idx] = 1.0

    # Handle Python dict (bit_idx: count mapping)
    elif isinstance(fingerprint, dict):
        for bit_idx, count in fingerprint.items():
            if 0 <= bit_idx < n_bits:
                dense[bit_idx] = float(count)

    # Handle list/tuple of on-bit indices
    elif isinstance(fingerprint, (list, tuple)):
        # Check if it's a list of indices or a full vector
        if len(fingerprint) == n_bits:
            # Full vector, return as-is after conversion
            return np.asarray(fingerprint, dtype=dtype)
        else:
            # List of on-bit indices
            for bit_idx in fingerprint:
                if 0 <= bit_idx < n_bits:
                    dense[bit_idx] = 1.0

    # Handle numpy array (already in correct format)
    elif isinstance(fingerprint, np.ndarray):
        if len(fingerprint) == n_bits:
            return fingerprint.astype(dtype)
        else:
            # Treat as list of indices
            for bit_idx in fingerprint:
                if 0 <= bit_idx < n_bits:
                    dense[bit_idx] = 1.0

    else:
        # Try to iterate as a sequence
        try:
            for bit_idx in fingerprint:
                if 0 <= bit_idx < n_bits:
                    dense[bit_idx] = 1.0
        except (TypeError, ValueError):
            raise ValueError(f"Unsupported fingerprint type: {type(fingerprint)}")

    return dense


def rdkit_sparse_to_csr(fingerprints, n_bits: int = 2048, dtype=np.float32) -> sparse.csr_matrix:
    """Convert RDKit sparse fingerprints to scipy CSR sparse matrix.

    Parameters
    ----------
    fingerprints : single fingerprint or list of fingerprints
        RDKit fingerprints in various formats.

    n_bits : int, default=2048
        Size of the fingerprint vectors.

    dtype : numpy dtype, default=np.float32
        Data type of the output matrix.

    Returns
    -------
    sparse.csr_matrix
        Sparse CSR matrix of shape (n_samples, n_bits).

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> mols = [Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CC')]
    >>> fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols]
    >>> csr_matrix = rdkit_sparse_to_csr(fps, n_bits=2048)
    """
    # Handle single fingerprint
    if not isinstance(fingerprints, (list, tuple, np.ndarray)):
        fingerprints = [fingerprints]
    elif isinstance(fingerprints, np.ndarray) and fingerprints.ndim == 1:
        # Could be a single dense fingerprint or array of fingerprints
        if len(fingerprints) == n_bits:
            fingerprints = [fingerprints]

    n_samples = len(fingerprints)
    rows, cols, data = [], [], []

    for i, fp in enumerate(fingerprints):
        if fp is None:
            continue

        # Extract on-bits and values
        if hasattr(fp, "GetOnBits"):
            # BitVect types
            for bit_idx in fp.GetOnBits():
                if 0 <= bit_idx < n_bits:
                    rows.append(i)
                    cols.append(bit_idx)
                    data.append(1.0)

        elif hasattr(fp, "GetNonzeroElements"):
            # SparseIntVect types
            for bit_idx, count in fp.GetNonzeroElements().items():
                if 0 <= bit_idx < n_bits:
                    rows.append(i)
                    cols.append(bit_idx)
                    data.append(float(count))

        elif isinstance(fp, set):
            for bit_idx in fp:
                if 0 <= bit_idx < n_bits:
                    rows.append(i)
                    cols.append(bit_idx)
                    data.append(1.0)

        elif isinstance(fp, dict):
            for bit_idx, count in fp.items():
                if 0 <= bit_idx < n_bits:
                    rows.append(i)
                    cols.append(bit_idx)
                    data.append(float(count))

        elif isinstance(fp, (list, tuple, np.ndarray)):
            if len(fp) == n_bits:
                # Full vector
                for j, val in enumerate(fp):
                    if val != 0:
                        rows.append(i)
                        cols.append(j)
                        data.append(float(val))
            else:
                # List of indices
                for bit_idx in fp:
                    if 0 <= bit_idx < n_bits:
                        rows.append(i)
                        cols.append(bit_idx)
                        data.append(1.0)

        else:
            # Try to iterate
            try:
                for bit_idx in fp:
                    if 0 <= bit_idx < n_bits:
                        rows.append(i)
                        cols.append(bit_idx)
                        data.append(1.0)
            except (TypeError, ValueError):
                raise ValueError(f"Unsupported fingerprint type: {type(fp)}")

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_bits), dtype=dtype)


def rdkit_sparse_to_csc(fingerprints, n_bits: int = 2048, dtype=np.float32) -> sparse.csc_matrix:
    """Convert RDKit sparse fingerprints to scipy CSC sparse matrix.

    Parameters
    ----------
    fingerprints : single fingerprint or list of fingerprints
        RDKit fingerprints in various formats.

    n_bits : int, default=2048
        Size of the fingerprint vectors.

    dtype : numpy dtype, default=np.float32
        Data type of the output matrix.

    Returns
    -------
    sparse.csc_matrix
        Sparse CSC matrix of shape (n_samples, n_bits).
    """
    csr = rdkit_sparse_to_csr(fingerprints, n_bits=n_bits, dtype=dtype)
    return csr.tocsc()


def rdkit_sparse_to_numpy(fingerprints, n_bits: int = 2048, dtype=np.float32) -> np.ndarray:
    """Convert RDKit sparse fingerprints to dense numpy array.

    Parameters
    ----------
    fingerprints : single fingerprint or list of fingerprints
        RDKit fingerprints in various formats.

    n_bits : int, default=2048
        Size of the fingerprint vectors.

    dtype : numpy dtype, default=np.float32
        Data type of the output array.

    Returns
    -------
    np.ndarray
        Dense numpy array of shape (n_samples, n_bits).

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> mols = [Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CC')]
    >>> fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols]
    >>> dense_matrix = rdkit_sparse_to_numpy(fps, n_bits=2048)
    """
    # Handle single fingerprint
    if not isinstance(fingerprints, (list, tuple)):
        fingerprints = [fingerprints]
    elif isinstance(fingerprints, np.ndarray) and fingerprints.ndim == 1:
        if len(fingerprints) == n_bits:
            fingerprints = [fingerprints]

    n_samples = len(fingerprints)
    dense_matrix = np.zeros((n_samples, n_bits), dtype=dtype)

    for i, fp in enumerate(fingerprints):
        dense_matrix[i] = rdkit_sparse_to_dense(fp, n_bits=n_bits, dtype=dtype)

    return dense_matrix


def rdkit_sparse_to_sklearn(
    fingerprints, n_bits: int = 2048, output_format: str = "auto", dtype=np.float32
) -> Union[np.ndarray, sparse.csr_matrix, sparse.csc_matrix]:
    """Convert RDKit sparse fingerprints to sklearn-compatible format.

    Parameters
    ----------
    fingerprints : single fingerprint or list of fingerprints
        RDKit fingerprints in various formats.

    n_bits : int, default=2048
        Size of the fingerprint vectors.

    output_format : {'auto', 'dense', 'csr', 'csc'}, default='auto'
        Output format:
        - 'auto': Choose based on sparsity (CSR if >90% sparse)
        - 'dense': Dense numpy array
        - 'csr': Compressed Sparse Row format
        - 'csc': Compressed Sparse Column format

    dtype : numpy dtype, default=np.float32
        Data type of the output.

    Returns
    -------
    array-like
        Fingerprints in sklearn-compatible format.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> from sklearn.naive_bayes import BernoulliNB
    >>>
    >>> mols = [Chem.MolFromSmiles(smi) for smi in ['CCO', 'CC', 'CCC']]
    >>> fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols]
    >>> X = rdkit_sparse_to_sklearn(fps, output_format='csr')
    >>> y = [0, 1, 0]
    >>>
    >>> clf = BernoulliNB()
    >>> clf.fit(X, y)
    """
    if output_format == "dense":
        return rdkit_sparse_to_numpy(fingerprints, n_bits=n_bits, dtype=dtype)
    elif output_format == "csr":
        return rdkit_sparse_to_csr(fingerprints, n_bits=n_bits, dtype=dtype)
    elif output_format == "csc":
        return rdkit_sparse_to_csc(fingerprints, n_bits=n_bits, dtype=dtype)
    elif output_format == "auto":
        # First convert to CSR to check sparsity
        csr_matrix = rdkit_sparse_to_csr(fingerprints, n_bits=n_bits, dtype=dtype)
        sparsity = 1.0 - (csr_matrix.nnz / (csr_matrix.shape[0] * csr_matrix.shape[1]))

        if sparsity > 0.9:  # More than 90% sparse
            return csr_matrix
        else:
            return csr_matrix.toarray()
    else:
        raise ValueError(f"Unknown output_format: {output_format}. Choose from 'auto', 'dense', 'csr', 'csc'.")


class RDKitFingerprintConverter:
    """Converter class for batch processing RDKit fingerprints.

    This class provides methods to convert RDKit fingerprints to various
    sklearn-compatible formats with caching and validation.

    Parameters
    ----------
    n_bits : int, default=2048
        Size of the fingerprint vectors.

    output_format : {'auto', 'dense', 'csr', 'csc'}, default='csr'
        Default output format for conversions. Default 'csr' for memory efficiency
        with molecular fingerprints which are typically very sparse.

    dtype : numpy dtype, default=np.float32
        Data type of the output.

    validate : bool, default=True
        Whether to validate input fingerprints.

    Attributes
    ----------
    n_features_ : int
        Number of features (bits) in the fingerprints.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>>
    >>> converter = RDKitFingerprintConverter(n_bits=2048, output_format='csr')
    >>>
    >>> mols = [Chem.MolFromSmiles(smi) for smi in ['CCO', 'CC', 'CCC']]
    >>> fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols]
    >>>
    >>> X = converter.convert(fps)
    >>> print(f"Shape: {X.shape}, Sparsity: {converter.get_sparsity(X):.2%}")
    """

    def __init__(self, n_bits: int = 2048, output_format: str = "csr", dtype=np.float32, validate: bool = True):
        self.n_bits = n_bits
        self.output_format = output_format
        self.dtype = dtype
        self.validate = validate
        self.n_features_ = n_bits

    def convert(
        self, fingerprints, output_format: Optional[str] = None
    ) -> Union[np.ndarray, sparse.csr_matrix, sparse.csc_matrix]:
        """Convert fingerprints to sklearn format.

        Parameters
        ----------
        fingerprints : single fingerprint or list of fingerprints
            RDKit fingerprints to convert.

        output_format : str, optional
            Override default output format for this conversion.

        Returns
        -------
        array-like
            Converted fingerprints.
        """
        if output_format is None:
            output_format = self.output_format

        if self.validate:
            self._validate_fingerprints(fingerprints)

        return rdkit_sparse_to_sklearn(fingerprints, n_bits=self.n_bits, output_format=output_format, dtype=self.dtype)

    def to_dense(self, fingerprints) -> np.ndarray:
        """Convert to dense numpy array."""
        return rdkit_sparse_to_numpy(fingerprints, self.n_bits, self.dtype)

    def to_csr(self, fingerprints) -> sparse.csr_matrix:
        """Convert to CSR sparse matrix."""
        return rdkit_sparse_to_csr(fingerprints, self.n_bits, self.dtype)

    def to_csc(self, fingerprints) -> sparse.csc_matrix:
        """Convert to CSC sparse matrix."""
        return rdkit_sparse_to_csc(fingerprints, self.n_bits, self.dtype)

    def _validate_fingerprints(self, fingerprints):
        """Validate that fingerprints are in a supported format."""
        if fingerprints is None:
            raise ValueError("Fingerprints cannot be None")

        # Check if it's a single fingerprint or a collection
        if not isinstance(fingerprints, (list, tuple, np.ndarray)):
            fingerprints = [fingerprints]

        for i, fp in enumerate(fingerprints):
            if fp is None:
                continue

            # Check for supported types
            valid = (
                hasattr(fp, "GetOnBits")
                or hasattr(fp, "GetNonzeroElements")
                or isinstance(fp, (set, dict, list, tuple, np.ndarray))
            )

            if not valid:
                # Try to iterate as last resort
                try:
                    iter(fp)
                except TypeError:
                    raise ValueError(f"Fingerprint at index {i} is not in a supported format. Got type: {type(fp)}")

    @staticmethod
    def get_sparsity(matrix) -> float:
        """Calculate sparsity of a matrix.

        Parameters
        ----------
        matrix : array-like
            Dense or sparse matrix.

        Returns
        -------
        float
            Sparsity ratio (fraction of zero elements).
        """
        if sparse.issparse(matrix):
            return 1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1]))
        else:
            return np.mean(matrix == 0)

    def get_statistics(self, fingerprints) -> Dict[str, Any]:
        """Get statistics about the fingerprints.

        Parameters
        ----------
        fingerprints : list of fingerprints
            RDKit fingerprints to analyze.

        Returns
        -------
        dict
            Statistics including sparsity, average on-bits, etc.
        """
        matrix = self.to_csr(fingerprints)

        stats = {
            "n_samples": matrix.shape[0],
            "n_features": matrix.shape[1],
            "sparsity": self.get_sparsity(matrix),
            "avg_on_bits": matrix.nnz / matrix.shape[0],
            "min_on_bits": min(matrix.getnnz(axis=1)),
            "max_on_bits": max(matrix.getnnz(axis=1)),
            "total_unique_bits": len(np.unique(matrix.nonzero()[1])),
        }

        return stats


# Convenience functions for direct use
def convert_fingerprints(
    fingerprints, n_bits: int = 2048, output_format: str = "csr", dtype=np.float32
) -> Union[np.ndarray, sparse.csr_matrix, sparse.csc_matrix]:
    """Convenience function to convert RDKit fingerprints to sklearn format.

    This is a simple wrapper around rdkit_sparse_to_sklearn for ease of use.

    Parameters
    ----------
    fingerprints : single fingerprint or list of fingerprints
        RDKit fingerprints in various formats.

    n_bits : int, default=2048
        Size of the fingerprint vectors.

    output_format : {'auto', 'dense', 'csr', 'csc'}, default='csr'
        Output format for the fingerprints. Default 'csr' for memory efficiency
        with molecular fingerprints which are typically very sparse.

    dtype : numpy dtype, default=np.float32
        Data type of the output.

    Returns
    -------
    array-like
        Fingerprints in sklearn-compatible format.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    >>> X = convert_fingerprints(fp)  # Returns sparse CSR matrix by default
    """
    return rdkit_sparse_to_sklearn(fingerprints, n_bits=n_bits, output_format=output_format, dtype=dtype)


class FingerprintTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for RDKit fingerprints.

    This transformer converts various RDKit fingerprint formats (sets, dicts,
    sparse representations) into dense or sparse matrices suitable for sklearn.
    Provides full sklearn pipeline compatibility with fit/transform interface.

    Parameters
    ----------
    n_bits : int, default=2048
        Number of bits in the fingerprint. Common values are 1024, 2048, 4096.

    output_format : {'auto', 'dense', 'csr', 'csc'}, default='csr'
        Output format for the transformed matrix:
        - 'csr': Compressed Sparse Row matrix (memory efficient)
        - 'csc': Compressed Sparse Column matrix
        - 'dense': Dense numpy array
        - 'auto': Automatically choose based on sparsity

    dtype : dtype, default=np.float32
        Data type of the output array.

    Attributes
    ----------
    n_features_out_ : int
        Number of output features (equal to n_bits).

    Examples
    --------
    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> from sklearn.pipeline import Pipeline
    >>> from laplaciannb import LaplacianNB, FingerprintTransformer
    >>>
    >>> # Generate fingerprints as sets of on-bits
    >>> mols = [Chem.MolFromSmiles('CCO'), Chem.MolFromSmiles('CC')]
    >>> fps = [set(AllChem.GetMorganFingerprintAsBitVect(mol, 2).GetOnBits())
    ...        for mol in mols]
    >>>
    >>> # Create sklearn pipeline
    >>> pipeline = Pipeline([
    ...     ('fingerprints', FingerprintTransformer(n_bits=2048)),
    ...     ('classifier', LaplacianNB())
    >>> ])
    >>>
    >>> # Use in cross-validation, grid search, etc.
    >>> y = [0, 1]
    >>> pipeline.fit(fps, y)
    """

    def __init__(self, n_bits=2048, output_format="csr", dtype=np.float32):
        self.n_bits = n_bits
        self.output_format = output_format
        self.dtype = dtype

    def fit(self, X, y=None):
        """Fit the transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input samples. Each sample can be:
            - A set of on-bit indices
            - A dictionary mapping bit indices to counts
            - A sparse fingerprint object (RDKit BitVect, etc.)
            - A list/tuple of on-bit indices

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_features_out_ = self.n_bits
        return self

    def transform(self, X):
        """Transform fingerprints to matrix format.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input samples in fingerprint format.

        Returns
        -------
        X_transformed : {ndarray, sparse matrix} of shape (n_samples, n_bits)
            Transformed fingerprint matrix.
        """
        check_is_fitted(self)

        # Use our existing conversion function
        return convert_fingerprints(X, n_bits=self.n_bits, output_format=self.output_format, dtype=self.dtype)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input samples in fingerprint format.

        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        X_transformed : {ndarray, sparse matrix} of shape (n_samples, n_bits)
            Transformed fingerprint matrix.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present for API consistency.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Array of feature names.
        """
        check_is_fitted(self)
        return np.array([f"bit_{i}" for i in range(self.n_bits)], dtype=object)
