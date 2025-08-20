#!/usr/bin/env python3
"""
Fingerprint Conversion Benchmark
===============================

Test the performance of rdkit_to_csr function with different dataset sizes
and parameters.
"""

import sys
import os

# Add src to path so we can import laplaciannb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from laplaciannb.fingerprint_utils import rdkit_to_csr, benchmark_fingerprint_conversion

def main():
    """Run fingerprint conversion benchmarks."""
    print("LaplacianNB Fingerprint Conversion Benchmark")
    print("=" * 50)
    
    try:
        # Quick test with small dataset
        print("\n1. Quick Test (50 molecules)")
        print("-" * 30)
        test_smiles = [
            "CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "CCCCCCCCCCCCCCCC", "CC1=CC=C(C=C1)C(=O)O"
        ] * 10  # 50 molecules
        
        X = rdkit_to_csr(test_smiles, radius=2, show_progress=True)
        print(f"✓ Successfully converted {X.shape[0]} molecules")
        
        # Medium test
        print("\n2. Medium Test (200 molecules)")
        print("-" * 30)
        medium_smiles = test_smiles * 4  # 200 molecules
        X_medium = rdkit_to_csr(medium_smiles, radius=2, show_progress=True)
        
        # Comprehensive benchmark
        print("\n3. Comprehensive Benchmark")
        print("-" * 30)
        benchmark_fingerprint_conversion(
            n_molecules=1000, 
            radii=[1, 2, 3],
            molecules_per_test=[100, 500, 1000]
        )
        
        print("\n" + "=" * 50)
        print("✓ All benchmarks completed successfully!")
        print("✓ LaplacianNB fingerprint conversion is ready for production")
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install rdkit scikit-learn scipy")
    except Exception as e:
        print(f"Error during benchmark: {e}")

if __name__ == "__main__":
    main()
