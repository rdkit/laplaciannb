import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import csr_matrix


def rdkit_to_csr(smiles_list, radius=2):
    """Convert RDKit sparse Morgan fingerprints to CSR matrix with lossless conversion."""
    row_ind = []
    col_ind = []

    # Create Morgan fingerprint generator
    mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)

    for i, mol in enumerate(mol_list):
        if mol is None:
            continue

        # Get sparse fingerprint
        sfp = mfpgen.GetSparseFingerprint(mol)
        for bit in set(sfp.GetOnBits()):
            # Reinterpret signed int32 as unsigned int32
            # This maps [-2^31, 2^31-1] to [0, 2^32-1] losslessly
            col_idx = np.uint32(bit & 0xFFFFFFFF)

            row_ind.append(i)
            col_ind.append(col_idx)
            data = np.ones(len(row_ind), dtype=np.bool)

    return csr_matrix((data, (row_ind, col_ind)), shape=(len(mol_list), 2**32), dtype=np.bool)
