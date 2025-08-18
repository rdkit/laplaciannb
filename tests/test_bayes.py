from pathlib import Path

# from bayes.bayes import get_fp
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from laplaciannb.legacy.LaplacianNB import LaplacianNB


def test_bayes():
    clf = LaplacianNB()
    rng = np.random.RandomState(1)
    arr = rng.randint(2, size=(6, 100))
    Y = np.array([1, 2, 3, 4, 4, 5])
    Xlist = []
    for i in arr:
        Xlist.append(set(i.nonzero()[0]))
    X = np.array(Xlist)
    clf.fit(X, Y)

    assert_array_equal(clf.feature_count_, [55.0, 46.0, 53.0, 90.0, 44.0])
    assert_array_equal(clf.class_count_, [1.0, 1.0, 1.0, 2.0, 1.0])
    assert clf.feature_all_ == 288


def test_lmnb_prior_unobserved_targets():
    # test smoothing of prior for yet unobserved targets

    # Create toy training data
    X = np.array([{1}, {0}])
    y = np.array([0, 1])

    clf = LaplacianNB()
    clf.fit(X, y)

    assert_array_equal(clf.predict(np.array([{1}])), np.array([0]))
    assert_array_equal(clf.predict(np.array([{0}])), np.array([1]))
    assert_array_equal(clf.predict(np.array([{0, 1}])), np.array([0]))


def test_rdkit():
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    from laplaciannb.legacy.LaplacianNB import LaplacianNB

    def get_fp(smiles: str) -> set:
        """Function to calculate MorganFingerprint from smiles.
        It returns index of all '1' bits of not-folded fingerprint.

        Args:
            smiles (str): smiles string

        Returns:
            set: return set of index of '1' bits.
        """

        mol = Chem.MolFromSmiles(smiles)
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        fp = mfpgen.GetSparseFingerprint(mol)
        return set(fp.GetOnBits())

    DATA_PATH = Path(__file__).parent.parent.joinpath("tests/data/")
    file = str(DATA_PATH.joinpath("smiles_test.csv"))
    df = pd.read_csv(file)
    df["sets"] = df["smiles"].apply(
        lambda x: get_fp(x),
    )
    X = df["sets"]
    y = df["activity"]
    clf = LaplacianNB()
    clf.fit(X, y)

    assert_array_equal(clf.feature_count_, [42727.0, 46838.0])
    assert_array_equal(clf.class_count_, [1000.0, 1000.0])
    assert clf.feature_all_ == 89565


def test_joint_log_likelihood():
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    from laplaciannb.legacy.LaplacianNB import LaplacianNB

    def get_fp(smiles: str) -> set:
        """Function to calculate MorganFingerprint from smiles.
        It returns index of all '1' bits of not-folded fingerprint.

        Args:
            smiles (str): smiles string

        Returns:
            set: return set of index of '1' bits.
        """

        mol = Chem.MolFromSmiles(smiles)
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        fp = mfpgen.GetSparseFingerprint(mol)
        return set(fp.GetOnBits())

    DATA_PATH = Path(__file__).parent.parent.joinpath("tests/data/")
    file = str(DATA_PATH.joinpath("smiles_test.csv"))
    df = pd.read_csv(file)
    df["sets"] = df["smiles"].apply(
        lambda x: get_fp(x),
    )
    X = df["sets"]
    y = df["activity"]
    clf = LaplacianNB()
    clf.fit(X, y)

    # check if algorithm can predict if index is out of range of fitted ones
    new_df = pd.DataFrame({"sets": [{10210210310210}]})
    new_X = new_df["sets"]
    try:
        clf._joint_log_likelihood(new_X)
    except Exception as exc:
        raise AssertionError(f"'_joint_log_likelihood' raised an exception {exc}")
