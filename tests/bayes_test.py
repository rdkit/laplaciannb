from pathlib import Path

# from bayes.bayes import get_fp
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from bayes.LaplacianNB import LaplacianNB


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
    from rdkit.Chem import AllChem

    from bayes.LaplacianNB import LaplacianNB

    def get_fp(smiles: str) -> dict:
        """Function to calculate MorganFingerprint from smiles.
        It returns index of all '1' bits of not-folded fingerprint.

        Args:
            smiles (str): smiles string

        Returns:
            dict: return dict of index of '1' bits.
        """

        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprint(mol, 2)
        return set(fp.GetNonzeroElements().keys())

    DATA_PATH = Path(__file__).parent.parent.joinpath("tests/data/")
    file = str(DATA_PATH.joinpath("smiles_test.csv"))
    df = pd.read_csv(file)
    df["dicts"] = df["smiles"].apply(
        lambda x: get_fp(x),
    )
    X = df["dicts"]
    y = df["activity"]
    clf = LaplacianNB()
    clf.fit(X, y)

    assert_array_equal(clf.feature_count_, [42727.0, 46838.0])
    assert_array_equal(clf.class_count_, [1000.0, 1000.0])
    assert clf.feature_all_ == 89565
