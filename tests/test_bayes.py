from pathlib import Path

# from bayes.bayes import get_fp
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from laplaciannb import LaplacianNB


def test_bayes():
    from scipy.sparse import csr_matrix

    clf = LaplacianNB()
    rng = np.random.RandomState(1)
    arr = rng.randint(2, size=(6, 100))
    Y = np.array([1, 2, 3, 4, 4, 5])

    # Convert binary array to CSR matrix
    X = csr_matrix(arr, dtype=np.bool_)
    clf.fit(X, Y)

    assert_array_equal(clf.feature_count_, [55.0, 46.0, 53.0, 90.0, 44.0])
    assert_array_equal(clf.class_count_, [1.0, 1.0, 1.0, 2.0, 1.0])
    assert clf.feature_all_ == 288


def test_lmnb_prior_unobserved_targets():
    # test smoothing of prior for yet unobserved targets
    from scipy.sparse import csr_matrix

    # Create toy training data as sparse matrices
    # First sample has feature 1, second sample has feature 0
    row = [0, 1]
    col = [1, 0]
    data = [1, 1]
    X = csr_matrix((data, (row, col)), shape=(2, 2), dtype=np.bool_)
    y = np.array([0, 1])

    clf = LaplacianNB()
    clf.fit(X, y)

    # Test predictions - ensure matrix dimensions match training data (2 features)
    test1 = csr_matrix(([1], ([0], [1])), shape=(1, 2), dtype=np.bool_)  # Feature 1 active
    test2 = csr_matrix(([1], ([0], [0])), shape=(1, 2), dtype=np.bool_)  # Feature 0 active
    test3 = csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(1, 2), dtype=np.bool_)  # Both features active

    assert_array_equal(clf.predict(test1), np.array([0]))
    assert_array_equal(clf.predict(test2), np.array([1]))
    assert_array_equal(clf.predict(test3), np.array([0]))


def test_rdkit():
    from laplaciannb import LaplacianNB
    from laplaciannb.fingerprint_utils import rdkit_to_csr

    DATA_PATH = Path(__file__).parent.parent.joinpath("tests/data/")
    file = str(DATA_PATH.joinpath("smiles_test.csv"))
    df = pd.read_csv(file)

    # Convert to sparse CSR matrix using our fingerprint utility
    X_sparse = rdkit_to_csr(df["smiles"].values, radius=2)

    y = df["activity"]
    clf = LaplacianNB()
    clf.fit(X_sparse, y)

    assert_array_equal(clf.feature_count_, [42727.0, 46838.0])
    assert_array_equal(clf.class_count_, [1000.0, 1000.0])
    assert clf.feature_all_ == 89565


def test_joint_log_likelihood():
    """Test joint log likelihood with CSR matrices."""
    from scipy.sparse import csr_matrix

    from laplaciannb import LaplacianNB
    from laplaciannb.fingerprint_utils import rdkit_to_csr

    DATA_PATH = Path(__file__).parent.parent.joinpath("tests/data/")
    file = str(DATA_PATH.joinpath("smiles_test.csv"))
    df = pd.read_csv(file)

    # Convert to CSR matrix using fingerprint utility
    X = rdkit_to_csr(df["smiles"].values, radius=2)
    y = df["activity"]
    clf = LaplacianNB()
    clf.fit(X, y)

    # Test with a feature index that might be out of range of fitted ones
    # Create a sparse matrix with a high but valid feature index
    test_row = [0]
    test_col = [2**30]  # Use a large but valid index within 2^32-1 limit
    test_data = [1]
    new_X = csr_matrix((test_data, (test_row, test_col)), shape=(1, 2**32 - 1), dtype=np.bool_)

    try:
        clf._joint_log_likelihood(new_X)
    except Exception as exc:
        raise AssertionError(f"'_joint_log_likelihood' raised an exception {exc}")


def test_csr_fingerprint_conversion():
    """Test the new CSR fingerprint conversion functionality."""
    from laplaciannb.fingerprint_utils import rdkit_to_csr

    # Create test molecules
    smiles_list = ["CCO", "CC", "CCC", "CCCC"]

    # Convert to CSR matrix
    X_sparse = rdkit_to_csr(smiles_list, radius=2)

    # Basic validation
    assert X_sparse.shape[0] == len(smiles_list)
    assert X_sparse.shape[1] == 2**32
    assert X_sparse.nnz > 0

    # Test that different molecules have different fingerprints
    fingerprint_rows = []
    for i in range(X_sparse.shape[0]):
        row = X_sparse[i]
        row_coo = row.tocoo()
        fingerprint_set = set(zip(row_coo.col, row_coo.data))
        fingerprint_rows.append(fingerprint_set)

    # Verify that molecules have some different features
    assert len({len(fp) for fp in fingerprint_rows}) > 1  # Different numbers of features

    print(f"Successfully created CSR matrix: {X_sparse.shape}, nnz: {X_sparse.nnz}")
    print(f"Fingerprint sizes: {[len(fp) for fp in fingerprint_rows]}")
