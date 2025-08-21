#!/usr/bin/env python3
"""
Simple LaplacianNB Example
=========================

A minimal example showing basic LaplacianNB usage with molecular data.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from laplaciannb import LaplacianNB
from laplaciannb.fingerprint_utils import rdkit_to_csr


# Sample molecular data
smiles = [
    "CCO",  # Ethanol - inactive
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin - active
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen - active
    "CCCCCCCCCCCCCCCC",  # Palmitic acid - inactive
    "CC1=CC=C(C=C1)C(=O)O",  # p-Toluic acid - active
]
y = [0, 1, 1, 0, 1]  # Activity labels (0=inactive, 1=active)

# Convert SMILES to sparse matrix
print("Converting molecular fingerprints...")
X = rdkit_to_csr(smiles, radius=2)
print(f"Matrix shape: {X.shape}")
print(f"Sparsity: {1 - X.nnz / (X.shape[0] * X.shape[1]):.6f}")

# Train classifier
print("\nTraining LaplacianNB...")
clf = LaplacianNB(alpha=1.0)
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)
probabilities = clf.predict_proba(X)

# Display results
print("\nResults:")
print("-" * 40)
for i, (smiles_str, true_label, pred_label, prob) in enumerate(zip(smiles, y, predictions, probabilities)):
    print(f"Molecule {i+1}: {smiles_str[:20]}")
    print(f"  True: {true_label}, Predicted: {pred_label}")
    print(f"  Probabilities: [Inactive: {prob[0]:.3f}, Active: {prob[1]:.3f}]")
    print()

# Calculate accuracy
accuracy = sum(predictions == y) / len(y)
print(f"Accuracy: {accuracy:.1%}")

# Advanced: Extract original fingerprint indices
print("\n" + "=" * 50)
print("EXTRACTING ORIGINAL FINGERPRINT INDICES")
print("=" * 50)

print("\nOriginal RDKit fingerprint indices for each molecule:")
print("-" * 50)


# Recreate the fingerprint generator to get individual fingerprints
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

for i, smiles_str in enumerate(smiles):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is not None:
        # Get sparse fingerprint with original indices
        sfp = mfpgen.GetSparseFingerprint(mol)
        original_indices = list(sfp.GetOnBits())

        # Convert to the same uint32 indices used in the matrix
        converted_indices = [int(np.uint32(bit & 0xFFFFFFFF)) for bit in original_indices]

        print(f"\nMolecule {i+1}: {smiles_str}")
        print(f"  Original indices: {original_indices[:10]}{'...' if len(original_indices) > 10 else ''}")
        print(f"  Converted indices: {converted_indices[:10]}{'...' if len(converted_indices) > 10 else ''}")
        print(f"  Total fingerprint bits: {len(original_indices)}")

# Show how to extract indices from the sparse matrix
print("\nExtracting indices from sparse matrix:")
print("-" * 50)

for i in range(X.shape[0]):
    # Get the column indices for row i
    start_idx = X.indptr[i]
    end_idx = X.indptr[i + 1]
    row_indices = X.indices[start_idx:end_idx]

    print(f"Molecule {i+1} active bits: {row_indices[:10]}{'...' if len(row_indices) > 10 else ''}")
    print(f"  Total: {len(row_indices)} active bits")

print("\n✓ You can now map back to original RDKit fingerprint indices")
print("✓ Useful for feature interpretation and chemical insights")

# Reverse mapping: From sparse matrix back to RDKit
print("\n" + "=" * 50)
print("REVERSE MAPPING: MATRIX → RDKIT")
print("=" * 50)

print("\nMapping sparse matrix indices back to original RDKit bits:")
print("-" * 50)


def uint32_to_rdkit_index(uint32_index):
    """Convert uint32 matrix index back to original RDKit signed int32."""
    # Convert back from unsigned to signed int32
    if uint32_index >= 2**31:
        return int(uint32_index) - 2**32
    else:
        return int(uint32_index)


# Example: Take the first molecule and show the reverse mapping
mol_idx = 0
print(f"\nExample with Molecule {mol_idx + 1}: {smiles[mol_idx]}")

# Get active indices from sparse matrix
start_idx = X.indptr[mol_idx]
end_idx = X.indptr[mol_idx + 1]
matrix_indices = X.indices[start_idx:end_idx]

print(f"Matrix indices (uint32): {matrix_indices}")

# Convert back to RDKit indices
rdkit_indices = [uint32_to_rdkit_index(idx) for idx in matrix_indices]
print(f"RDKit indices (int32):   {rdkit_indices}")

# Verify this matches the original fingerprint
mol = Chem.MolFromSmiles(smiles[mol_idx])
sfp = mfpgen.GetSparseFingerprint(mol)
original_indices = sorted(sfp.GetOnBits())
recovered_indices = sorted(rdkit_indices)

print(f"Original RDKit indices: {original_indices}")
print(f"Recovered indices:      {recovered_indices}")
print(f"Match: {'✓' if original_indices == recovered_indices else '✗'}")
