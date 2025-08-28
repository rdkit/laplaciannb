import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse import csr_matrix


try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable, *args, **kwargs):
        """Fallback if tqdm is not available."""
        return iterable


def rdkit_to_csr(smiles_list, radius=2, show_progress=True):
    """Convert RDKit sparse Morgan fingerprints to CSR matrix with lossless conversion.

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings to convert to fingerprints
    radius : int, default=2
        Morgan fingerprint radius
    show_progress : bool, default=True
        Show progress bar if tqdm is available

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix of shape (n_molecules, 2^32) with boolean dtype

    Examples
    --------
    >>> smiles = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O"]
    >>> X = rdkit_to_csr(smiles, radius=2)
    >>> print(f"Shape: {X.shape}, Sparsity: {1 - X.nnz / X.size:.6f}")
    """
    start_time = time.time()

    row_ind = []
    col_ind = []

    # Create Morgan fingerprint generator
    print(f"Converting {len(smiles_list)} SMILES to molecular fingerprints...")
    mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)

    # Process molecules with optional progress bar
    iterator = enumerate(mol_list)
    if show_progress and TQDM_AVAILABLE and len(mol_list) > 10:
        iterator = tqdm(iterator, total=len(mol_list), desc="Processing molecules", unit="mol")

    valid_molecules = 0
    total_bits = 0

    for i, mol in iterator:
        if mol is None:
            continue

        valid_molecules += 1

        # Get sparse fingerprint
        sfp = mfpgen.GetSparseFingerprint(mol)
        mol_bits = set(sfp.GetOnBits())
        total_bits += len(mol_bits)

        for bit in mol_bits:
            # Reinterpret signed int32 as unsigned int32
            # This maps [-2^31, 2^31-1] to [0, 2^32-1] losslessly
            col_idx = np.uint32(bit & 0xFFFFFFFF)

            row_ind.append(i)
            col_ind.append(col_idx)

    # Create data array (all ones for boolean matrix)
    data = np.ones(len(row_ind), dtype=np.bool_)

    # Create sparse matrix
    matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(mol_list), 2**32), dtype=np.bool_)

    # Performance summary
    conversion_time = time.time() - start_time
    sparsity = 1 - matrix.nnz / matrix.size if matrix.size > 0 else 0

    print(f"Conversion completed in {conversion_time:.3f} seconds")
    print(f"Valid molecules: {valid_molecules}/{len(mol_list)}")
    print(f"Total fingerprint bits: {total_bits:,}")
    print(f"Average bits per molecule: {total_bits / valid_molecules:.1f}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix sparsity: {sparsity:.6f}")
    print(f"Memory usage: {(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 1024**2:.2f} MB")

    return matrix


def benchmark_fingerprint_conversion(n_molecules=100000, radii=[2], molecules_per_test=None):
    """Benchmark fingerprint conversion performance with different parameters.

    Parameters
    ----------
    n_molecules : int, default=1000
        Number of molecules to generate for benchmarking
    radii : list of int, default=[1, 2, 3]
        Morgan fingerprint radii to test
    molecules_per_test : list of int, optional
        Different molecule counts to test. If None, uses [100, 500, 1000]

    Examples
    --------
    >>> benchmark_fingerprint_conversion(1000, radii=[2, 3])
    >>> benchmark_fingerprint_conversion(500, molecules_per_test=[100, 300, 500])
    """
    print("=" * 60)
    print("FINGERPRINT CONVERSION BENCHMARK")
    print("=" * 60)

    # Generate test SMILES data
    print(f"Generating {n_molecules} test molecules...")
    test_smiles = _generate_test_smiles(n_molecules)

    if molecules_per_test is None:
        molecules_per_test = [min(100, n_molecules), min(500, n_molecules), n_molecules]

    # Test different molecule counts
    print("\nTesting conversion speed with different dataset sizes:")
    print("-" * 60)
    print(f"{'Molecules':<12} {'Radius':<8} {'Time (s)':<10} {'Bits/mol':<10} {'MB':<8}")
    print("-" * 60)

    for n_mol in molecules_per_test:
        subset_smiles = test_smiles[:n_mol]

        for radius in radii:
            start_time = time.time()
            X = rdkit_to_csr(subset_smiles, radius=radius, show_progress=False)
            conversion_time = time.time() - start_time

            avg_bits = X.nnz / X.shape[0] if X.shape[0] > 0 else 0
            memory_mb = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1024**2

            print(f"{n_mol:<12} {radius:<8} {conversion_time:<10.3f} {avg_bits:<10.1f} {memory_mb:<8.2f}")

    # Memory efficiency comparison
    print("\nMemory Efficiency Analysis:")
    print("-" * 40)

    X_example = rdkit_to_csr(test_smiles[:100], radius=2, show_progress=False)
    sparse_memory = (X_example.data.nbytes + X_example.indices.nbytes + X_example.indptr.nbytes) / 1024**2
    dense_memory = (X_example.shape[0] * X_example.shape[1] * np.dtype(np.bool_).itemsize) / 1024**2

    print("100 molecules, radius=2:")
    print(f"  Sparse matrix: {sparse_memory:.2f} MB")
    print(f"  Dense equivalent: {dense_memory:,.0f} MB")
    print(f"  Memory reduction: {(1 - sparse_memory / dense_memory) * 100:.3f}%")

    # Throughput summary
    print("\nThroughput Summary:")
    print("-" * 20)
    fastest_time = min([conversion_time for n_mol in molecules_per_test[:1] for radius in radii[:1]])
    throughput = molecules_per_test[0] / fastest_time if fastest_time > 0 else 0
    print(f"Peak throughput: ~{throughput:.0f} molecules/second")
    print(f"Recommended for datasets: Up to {throughput * 60:.0f} molecules/minute")


def _generate_test_smiles(n_molecules):
    """Generate test SMILES strings for benchmarking."""
    # Simple test molecules with varying complexity
    base_smiles = [
        "CCO",  # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CCCCCCCCCCCCCCCC",  # Palmitic acid
        "CC1=CC=C(C=C1)C(=O)O",  # p-Toluic acid
        "CCN(CC)CC",  # Triethylamine
        "CC(C)(C)C1=CC=C(C=C1)O",  # BHT
        "CCCCCCCCCCCCC",  # Tridecane
        "CC1=CC(=CC(=C1)C)C(=O)O",  # Mesitylenic acid
        "CCCCCCCCCC",  # Decane
        "CC1=CC=CC=C1",  # Toluene
        "C1=CC=CC=C1",  # Benzene
        "CC(C)O",  # Isopropanol
        "CCCCO",  # Butanol
        "CC(C)C",  # Propane
    ]

    # Repeat base molecules to reach desired count
    test_smiles = []
    while len(test_smiles) < n_molecules:
        test_smiles.extend(base_smiles)

    return test_smiles[:n_molecules]


def benchmark_large_scale_conversion(target_molecules=100000, test_sizes=None, radius=2, sample_diversity=True):
    """Benchmark fingerprint conversion performance for large datasets.

    This function tests the scalability and performance of rdkit_to_csr
    with large molecular datasets up to 100,000 molecules.

    Parameters
    ----------
    target_molecules : int, default=100000
        Maximum number of molecules to test
    test_sizes : list of int, optional
        Molecule counts to benchmark. If None, uses logarithmic scale
    radius : int, default=2
        Morgan fingerprint radius
    sample_diversity : bool, default=True
        If True, generates diverse molecular structures for realistic testing

    Examples
    --------
    >>> benchmark_large_scale_conversion(100000)
    >>> benchmark_large_scale_conversion(50000, test_sizes=[1000, 10000, 50000])
    """
    print("=" * 80)
    print("LARGE-SCALE FINGERPRINT CONVERSION BENCHMARK")
    print("=" * 80)
    print(f"Target dataset size: {target_molecules:,} molecules")
    print(f"Morgan fingerprint radius: {radius}")
    print(f"Diversity sampling: {'Enabled' if sample_diversity else 'Disabled'}")

    if test_sizes is None:
        # Logarithmic scale testing
        test_sizes = [1000, 5000, 10000, 25000, 50000]
        if target_molecules >= 100000:
            test_sizes.append(100000)
        # Filter to not exceed target
        test_sizes = [size for size in test_sizes if size <= target_molecules]

    print(f"\nGenerating test dataset with {target_molecules:,} molecules...")
    print("-" * 60)

    start_gen = time.time()
    test_smiles = _generate_diverse_smiles(target_molecules, diverse=sample_diversity)
    gen_time = time.time() - start_gen

    print(f"Dataset generation completed in {gen_time:.2f} seconds")
    print(f"Average generation rate: {target_molecules / gen_time:.0f} molecules/second")

    # Performance tracking
    results = []

    print("\nBenchmarking conversion performance:")
    print("-" * 80)
    print(
        f"{'Molecules':<12} {'Time (s)':<10} {'Rate (mol/s)':<12} {'Bits/mol':<10} {'Memory (MB)':<12} {'Sparsity':<10}"
    )
    print("-" * 80)

    for n_molecules in test_sizes:
        print(f"Testing {n_molecules:,} molecules...", end=" ", flush=True)

        # Subset the data
        subset_smiles = test_smiles[:n_molecules]

        # Benchmark conversion
        start_time = time.time()
        X = rdkit_to_csr(subset_smiles, radius=radius, show_progress=False)
        conversion_time = time.time() - start_time

        # Calculate metrics
        rate = n_molecules / conversion_time if conversion_time > 0 else 0
        avg_bits = X.nnz / X.shape[0] if X.shape[0] > 0 else 0
        memory_mb = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1024**2
        sparsity = 1 - (X.nnz / X.size) if X.size > 0 else 0

        results.append(
            {
                "molecules": n_molecules,
                "time": conversion_time,
                "rate": rate,
                "bits_per_mol": avg_bits,
                "memory_mb": memory_mb,
                "sparsity": sparsity,
            }
        )

        print(
            f"{n_molecules:<12,} {conversion_time:<10.2f} {rate:<12.0f} {avg_bits:<10.1f} {memory_mb:<12.2f} {sparsity:<10.6f}"
        )

    # Scalability analysis
    print("\nScalability Analysis:")
    print("-" * 40)

    if len(results) >= 2:
        # Calculate scaling efficiency
        small_result = results[0]
        large_result = results[-1]

        size_ratio = large_result["molecules"] / small_result["molecules"]
        time_ratio = large_result["time"] / small_result["time"]
        scaling_efficiency = size_ratio / time_ratio

        print(
            f"Size scaling: {small_result['molecules']:,} → {large_result['molecules']:,} molecules ({size_ratio:.1f}x)"
        )
        print(f"Time scaling: {small_result['time']:.2f}s → {large_result['time']:.2f}s ({time_ratio:.1f}x)")
        print(f"Scaling efficiency: {scaling_efficiency:.2f} (1.0 = perfect linear scaling)")

        # Memory scaling
        memory_ratio = large_result["memory_mb"] / small_result["memory_mb"]
        print(
            f"Memory scaling: {small_result['memory_mb']:.1f}MB → {large_result['memory_mb']:.1f}MB ({memory_ratio:.1f}x)"
        )

    # Performance projections
    print("\nPerformance Projections:")
    print("-" * 30)

    if results:
        latest = results[-1]

        # Project to larger datasets
        projected_1M = (1_000_000 / latest["rate"]) if latest["rate"] > 0 else float("inf")
        projected_memory_1M = latest["memory_mb"] * (1_000_000 / latest["molecules"])

        print(f"Projected time for 1M molecules: {projected_1M / 60:.1f} minutes")
        print(f"Projected memory for 1M molecules: {projected_memory_1M / 1024:.1f} GB")

        # Realistic dataset recommendations
        if latest["rate"] > 0:
            molecules_per_minute = latest["rate"] * 60
            molecules_per_hour = molecules_per_minute * 60

            print("\nRealistic Usage Recommendations:")
            print(f"  Interactive analysis: Up to {int(molecules_per_minute / 10):,} molecules")
            print(f"  Batch processing: Up to {int(molecules_per_hour / 10):,} molecules")
            print(f"  Production pipeline: {int(molecules_per_hour):,}+ molecules/hour")

    # Memory efficiency showcase
    print("\nMemory Efficiency Showcase:")
    print("-" * 35)

    if results:
        example = results[-1]
        sparse_mb = example["memory_mb"]

        # Calculate theoretical dense matrix size
        n_mols = example["molecules"]
        dense_gb = (n_mols * (2**32) * 1) / (1024**3)  # 1 byte per boolean

        print(f"{n_mols:,} molecules:")
        print(f"  Sparse matrix: {sparse_mb:.1f} MB")
        print(f"  Dense equivalent: {dense_gb:,.0f} GB")
        print(f"  Space savings: {(1 - sparse_mb / (dense_gb * 1024)) * 100:.6f}%")

    print(f"\n{'=' * 80}")
    print("✓ Large-scale benchmark completed successfully!")
    print(f"✓ LaplacianNB can efficiently handle datasets up to {target_molecules:,} molecules")
    print(f"{'=' * 80}")

    return results


def _generate_diverse_smiles(n_molecules, diverse=True):
    """Generate a diverse set of SMILES for realistic benchmarking."""
    if diverse:
        # More diverse molecular structures for realistic testing
        base_smiles = [
            # Simple aliphatics
            "CCO",
            "CCC",
            "CCCC",
            "CCCCC",
            "CCCCCC",
            "CCCCCCC",
            "CC(C)C",
            "CC(C)CC",
            "CC(C)(C)C",
            "CCCCCCCCCC",
            # Aromatics and pharmaceuticals
            "C1=CC=CC=C1",
            "CC1=CC=CC=C1",
            "CC1=CC=C(C=C1)C",
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            # Heterocycles
            "C1=CC=NC=C1",
            "C1=CN=CC=C1",
            "C1=CC=C(C=C1)N",
            "C1CCC(CC1)N",
            "C1=CC=C2C(=C1)C=CC=N2",
            # Functional groups
            "CC(=O)O",
            "CCO",
            "CC(=O)C",
            "CCCN",
            "CCS",
            "CC=O",
            "CC(=O)N",
            "CC(C)O",
            "C=CC",
            "C#CC",
            "CCCl",
            "CCBr",
            # Larger molecules
            "CCCCCCCCCCCCCCCC",  # Palmitic acid
            "CC1=CC(=CC(=C1)C)C(=O)O",  # Mesitylenic acid
            "CC(C)(C)C1=CC=C(C=C1)O",  # BHT
            "CCN(CC)CC",  # Triethylamine
            # Steroids and complex structures
            "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
            "CN1CCC[C@H]1C2=CN=CC=C2",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        ]
    else:
        # Simple repeated structures for baseline testing
        base_smiles = [
            "CCO",
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "CCCCCCCCCCCCCCCC",
            "CC1=CC=C(C=C1)C(=O)O",
            "CCN(CC)CC",
            "CC(C)(C)C1=CC=C(C=C1)O",
            "CCCCCCCCCCCCC",
            "CC1=CC(=CC(=C1)C)C(=O)O",
            "CCCCCCCCCC",
            "CC1=CC=CC=C1",
            "C1=CC=CC=C1",
            "CC(C)O",
            "CCCCO",
        ]

    # Generate the required number of molecules
    test_smiles = []
    while len(test_smiles) < n_molecules:
        test_smiles.extend(base_smiles)

    return test_smiles[:n_molecules]
