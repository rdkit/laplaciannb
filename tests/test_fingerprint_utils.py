"""Tests for fingerprint utility functions."""

import numpy as np
import pytest
from scipy import sparse

from laplaciannb.fingerprint_utils import (
    RDKitFingerprintConverter,
    convert_fingerprints,
    rdkit_sparse_to_csr,
    rdkit_sparse_to_dense,
    rdkit_sparse_to_numpy,
    rdkit_sparse_to_sklearn,
)


class TestFingerprintUtils:
    """Test suite for fingerprint utility functions."""

    @pytest.fixture
    def sample_set_fingerprints(self):
        """Create sample fingerprints as sets (similar to RDKit on-bits)."""
        fps = [
            {1, 5, 10, 15, 20},
            {2, 6, 11, 16, 21},
            {1, 3, 7, 12, 17},
            set(),  # Empty fingerprint
        ]
        return fps

    @pytest.fixture
    def sample_dict_fingerprints(self):
        """Create sample fingerprints as dictionaries (count data)."""
        fps = [
            {1: 2, 5: 1, 10: 3},
            {2: 1, 6: 2, 11: 1},
            {1: 1, 3: 1, 7: 2},
        ]
        return fps

    def test_rdkit_sparse_to_dense_sets(self, sample_set_fingerprints):
        """Test conversion of set fingerprints to dense arrays."""
        n_bits = 25

        for fp in sample_set_fingerprints:
            dense = rdkit_sparse_to_dense(fp, n_bits=n_bits)

            assert dense.shape == (n_bits,)
            assert dense.dtype == np.float32

            # Check that on-bits are correctly set
            for bit_idx in fp:
                assert dense[bit_idx] == 1.0

            # Check that off-bits are zero
            off_bits = set(range(n_bits)) - fp
            for bit_idx in off_bits:
                assert dense[bit_idx] == 0.0

    def test_rdkit_sparse_to_dense_dicts(self, sample_dict_fingerprints):
        """Test conversion of dict fingerprints to dense arrays."""
        n_bits = 25

        for fp in sample_dict_fingerprints:
            dense = rdkit_sparse_to_dense(fp, n_bits=n_bits)

            assert dense.shape == (n_bits,)

            # Check that counts are correctly set
            for bit_idx, count in fp.items():
                assert dense[bit_idx] == float(count)

    def test_rdkit_sparse_to_csr(self, sample_set_fingerprints):
        """Test conversion to CSR sparse matrix."""
        n_bits = 25
        csr_matrix = rdkit_sparse_to_csr(sample_set_fingerprints, n_bits=n_bits)

        assert csr_matrix.shape == (len(sample_set_fingerprints), n_bits)
        assert sparse.issparse(csr_matrix)
        assert csr_matrix.format == "csr"

        # Convert back to dense for verification
        dense = csr_matrix.toarray()

        for i, fp in enumerate(sample_set_fingerprints):
            for bit_idx in fp:
                assert dense[i, bit_idx] == 1.0

    def test_rdkit_sparse_to_numpy(self, sample_set_fingerprints):
        """Test conversion to dense numpy array."""
        n_bits = 25
        dense_matrix = rdkit_sparse_to_numpy(sample_set_fingerprints, n_bits=n_bits)

        assert dense_matrix.shape == (len(sample_set_fingerprints), n_bits)
        assert isinstance(dense_matrix, np.ndarray)

        for i, fp in enumerate(sample_set_fingerprints):
            for bit_idx in fp:
                assert dense_matrix[i, bit_idx] == 1.0

    def test_rdkit_sparse_to_sklearn_auto(self, sample_set_fingerprints):
        """Test auto format selection."""
        n_bits = 2048  # Large enough to trigger sparse format
        result = rdkit_sparse_to_sklearn(sample_set_fingerprints, n_bits=n_bits, output_format="auto")

        # Should be sparse due to high sparsity
        assert sparse.issparse(result)

    def test_rdkit_sparse_to_sklearn_dense(self, sample_set_fingerprints):
        """Test dense format selection."""
        n_bits = 25
        result = rdkit_sparse_to_sklearn(sample_set_fingerprints, n_bits=n_bits, output_format="dense")

        assert isinstance(result, np.ndarray)
        assert result.shape == (len(sample_set_fingerprints), n_bits)

    def test_convert_fingerprints_convenience(self, sample_set_fingerprints):
        """Test the convenience function."""
        n_bits = 25
        result = convert_fingerprints(sample_set_fingerprints, n_bits=n_bits)

        assert result.shape[0] == len(sample_set_fingerprints)
        assert result.shape[1] == n_bits

    def test_single_fingerprint_conversion(self):
        """Test conversion of a single fingerprint."""
        fp = {1, 5, 10}
        n_bits = 15

        dense = rdkit_sparse_to_dense(fp, n_bits=n_bits)
        assert dense.shape == (n_bits,)

        csr = rdkit_sparse_to_csr(fp, n_bits=n_bits)
        assert csr.shape == (1, n_bits)

        numpy_result = rdkit_sparse_to_numpy(fp, n_bits=n_bits)
        assert numpy_result.shape == (1, n_bits)

    def test_empty_fingerprint_handling(self):
        """Test handling of empty fingerprints."""
        empty_fps = [set(), {}, []]
        n_bits = 10

        for fp in empty_fps:
            dense = rdkit_sparse_to_dense(fp, n_bits=n_bits)
            assert np.all(dense == 0)

            csr = rdkit_sparse_to_csr([fp], n_bits=n_bits)
            assert csr.nnz == 0

    def test_list_fingerprints(self):
        """Test handling of list-based fingerprints."""
        # List of indices
        fp_indices = [1, 5, 10, 15]
        n_bits = 20

        dense = rdkit_sparse_to_dense(fp_indices, n_bits=n_bits)
        for idx in fp_indices:
            assert dense[idx] == 1.0

        # Full vector
        fp_full = np.zeros(n_bits)
        fp_full[[1, 5, 10]] = 1
        dense_full = rdkit_sparse_to_dense(fp_full, n_bits=n_bits)
        np.testing.assert_array_equal(dense_full, fp_full)

    def test_bounds_checking(self):
        """Test that out-of-bounds indices are ignored."""
        fp = {1, 5, 100}  # 100 is out of bounds
        n_bits = 10

        dense = rdkit_sparse_to_dense(fp, n_bits=n_bits)
        assert dense[1] == 1.0
        assert dense[5] == 1.0
        # Index 100 should be ignored, no error raised


class TestRDKitFingerprintConverter:
    """Test suite for the RDKitFingerprintConverter class."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return RDKitFingerprintConverter(n_bits=50, output_format="dense")

    @pytest.fixture
    def sample_fingerprints(self):
        """Sample fingerprints for testing."""
        return [
            {1, 5, 10, 15},
            {2, 6, 11, 16},
            {1, 3, 7, 12},
        ]

    def test_converter_initialization(self):
        """Test converter initialization."""
        converter = RDKitFingerprintConverter(n_bits=1024, output_format="csr", dtype=np.int32)

        assert converter.n_bits == 1024
        assert converter.output_format == "csr"
        assert converter.dtype == np.int32
        assert converter.n_features_ == 1024

    def test_converter_convert_method(self, converter, sample_fingerprints):
        """Test the convert method."""
        result = converter.convert(sample_fingerprints)

        assert isinstance(result, np.ndarray)  # Dense format
        assert result.shape == (len(sample_fingerprints), converter.n_bits)

    def test_converter_to_dense(self, converter, sample_fingerprints):
        """Test to_dense method."""
        result = converter.to_dense(sample_fingerprints)
        assert isinstance(result, np.ndarray)
        assert result.shape == (len(sample_fingerprints), converter.n_bits)

    def test_converter_to_csr(self, converter, sample_fingerprints):
        """Test to_csr method."""
        result = converter.to_csr(sample_fingerprints)
        assert sparse.issparse(result)
        assert result.format == "csr"

    def test_converter_to_csc(self, converter, sample_fingerprints):
        """Test to_csc method."""
        result = converter.to_csc(sample_fingerprints)
        assert sparse.issparse(result)
        assert result.format == "csc"

    def test_converter_get_sparsity(self, converter):
        """Test sparsity calculation."""
        # Dense matrix
        dense = np.array([[1, 0, 0], [0, 1, 0]])
        sparsity = converter.get_sparsity(dense)
        assert abs(sparsity - 2 / 3) < 1e-10  # 4 zeros out of 6 elements

        # Sparse matrix
        sparse_matrix = sparse.csr_matrix([[1, 0, 0], [0, 1, 0]])
        sparsity_sparse = converter.get_sparsity(sparse_matrix)
        assert abs(sparsity_sparse - 2 / 3) < 1e-10

    def test_converter_get_statistics(self, converter, sample_fingerprints):
        """Test statistics calculation."""
        stats = converter.get_statistics(sample_fingerprints)

        assert "n_samples" in stats
        assert "n_features" in stats
        assert "sparsity" in stats
        assert "avg_on_bits" in stats
        assert "min_on_bits" in stats
        assert "max_on_bits" in stats
        assert "total_unique_bits" in stats

        assert stats["n_samples"] == len(sample_fingerprints)
        assert stats["n_features"] == converter.n_bits

    def test_converter_validation_error(self, converter):
        """Test validation error handling."""
        # Should raise error for unsupported type (integer is not iterable)
        with pytest.raises(ValueError):
            converter._validate_fingerprints(42)

    def test_format_override(self, converter, sample_fingerprints):
        """Test output format override."""
        # Converter default is 'dense', but override to 'csr'
        result = converter.convert(sample_fingerprints, output_format="csr")
        assert sparse.issparse(result)
        assert result.format == "csr"


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_none_fingerprint(self):
        """Test handling of None fingerprint."""
        result = rdkit_sparse_to_dense(None, n_bits=10)
        assert np.all(result == 0)

        result_list = rdkit_sparse_to_numpy([None, {1, 2}], n_bits=10)
        assert result_list.shape == (2, 10)
        assert np.all(result_list[0] == 0)
        assert result_list[1, 1] == 1

    def test_invalid_format_error(self):
        """Test error for invalid output format."""
        with pytest.raises(ValueError, match="Unknown output_format"):
            rdkit_sparse_to_sklearn([{1, 2}], output_format="invalid")

    def test_unsupported_type_error(self):
        """Test error for completely unsupported types."""
        with pytest.raises(ValueError, match="Unsupported fingerprint type"):
            rdkit_sparse_to_dense(42)  # Integer is not supported

    def test_mixed_fingerprint_types(self):
        """Test handling of mixed fingerprint types."""
        mixed_fps = [
            {1, 2, 3},  # set
            {4: 1, 5: 2},  # dict
            [6, 7, 8],  # list
        ]

        result = rdkit_sparse_to_numpy(mixed_fps, n_bits=10)
        assert result.shape == (3, 10)

        # Check each type was handled correctly
        assert result[0, 1] == 1  # set
        assert result[1, 4] == 1  # dict
        assert result[2, 6] == 1  # list


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
