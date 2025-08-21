import numpy as np
from rdkit import Chem

from laplaciannb.fingerprint_utils import rdkit_to_csr


def csr_to_rdkit_bit(col_idx):
    """Convert CSR column index back to RDKit bit"""
    return np.int32(col_idx)


def get_test_molecules():
    """Get simple test molecules"""
    smiles = ["CCO", "CC", "CCC"]  # ethanol, methane, propane
    return [Chem.MolFromSmiles(smi) for smi in smiles]


class TestFingerprintCSRConversion:
    def test_rdkit_to_csr_basic(self):
        """Test basic RDKit to CSR conversion"""
        smiles = ["CCO", "CC", "CCC"]
        csr_matrix_result = rdkit_to_csr(smiles)

        # Basic checks
        assert csr_matrix_result.shape[0] == len(smiles)
        assert csr_matrix_result.shape[1] == 2**32
        assert csr_matrix_result.nnz > 0  # Should have non-zero elements

    def test_fingerprint_consistency(self):
        """Test that CSR conversion preserves fingerprint information"""
        smiles = ["CCO", "CC", "CCC"]
        csr_result = rdkit_to_csr(smiles)

        # Calculate total expected fingerprint bits across all molecules
        # Use the same API as the function
        from rdkit.Chem import rdFingerprintGenerator

        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

        total_expected_bits = 0
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                sfp = mfpgen.GetSparseFingerprint(mol)
                total_expected_bits += sfp.GetNumOnBits()

        # Check that we have the same total number of features
        assert csr_result.nnz == total_expected_bits

    def test_bit_conversion_roundtrip(self):
        """Test that bit conversion works both ways (WILL FAIL)"""
        # Test a few example bits
        test_bits = [-1000, 0, 1000]

        for original_bit in test_bits:
            # This will fail because mock just returns the same value
            recovered_bit = csr_to_rdkit_bit(original_bit)
            # For negative bits, this should fail with current mock
            assert recovered_bit == original_bit
