"""Example demonstrating LaplacianNB with fingerprint utilities."""

import numpy as np
from laplaciannb import LaplacianNB, convert_fingerprints, RDKitFingerprintConverter

# Example 1: Basic usage with set data
print("=== Example 1: Basic LaplacianNB Usage ===")
X_sets = np.array([
    {1, 5, 10, 15, 20},      # Sample 1: bits 1,5,10,15,20 are on
    {2, 6, 11, 16, 21},      # Sample 2: bits 2,6,11,16,21 are on  
    {1, 3, 7, 12, 17},       # Sample 3: bits 1,3,7,12,17 are on
    {4, 8, 13, 18, 22},      # Sample 4: bits 4,8,13,18,22 are on
], dtype=object)
y = np.array([0, 1, 0, 1])  # Binary classification

# Train classifier
clf = LaplacianNB()
clf.fit(X_sets, y)

# Predictions
predictions = clf.predict(X_sets)
probabilities = clf.predict_proba(X_sets)

print(f"Predictions: {predictions}")
print(f"Probabilities shape: {probabilities.shape}")
print(f"Sample probability for class 0: {probabilities[0, 0]:.3f}")

# Example 2: Using fingerprint conversion utilities
print("\n=== Example 2: Fingerprint Conversion ===")

# Simulate molecular fingerprints as sets of on-bits
molecular_fps = [
    {1, 5, 10, 15, 100, 200},    # Molecule 1
    {2, 6, 11, 16, 101, 201},    # Molecule 2  
    {1, 3, 7, 12, 102, 202},     # Molecule 3
    {4, 8, 13, 18, 103, 203},    # Molecule 4
]

# Convert to different formats
dense_matrix = convert_fingerprints(molecular_fps, n_bits=512, output_format='dense')
sparse_matrix = convert_fingerprints(molecular_fps, n_bits=512, output_format='csr')

print(f"Dense matrix shape: {dense_matrix.shape}")
print(f"Sparse matrix shape: {sparse_matrix.shape}")
print(f"Sparse matrix format: {sparse_matrix.format}")
print(f"Sparsity: {1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.3f}")

# Example 3: Using the converter class
print("\n=== Example 3: RDKitFingerprintConverter ===")

converter = RDKitFingerprintConverter(
    n_bits=1024,
    output_format='auto',  # Automatically choose based on sparsity
    dtype=np.float32
)

# Convert fingerprints
X_converted = converter.convert(molecular_fps)
stats = converter.get_statistics(molecular_fps)

print(f"Converted matrix type: {type(X_converted)}")
print(f"Matrix shape: {X_converted.shape}")
print(f"Statistics:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.3f}")
    else:
        print(f"  {key}: {value}")

# Train classifier with converted data
# Note: LaplacianNB expects sets of indices, so we need to convert back
def sparse_to_sets(sparse_matrix):
    """Convert sparse matrix back to array of sets for LaplacianNB."""
    sets = []
    for i in range(sparse_matrix.shape[0]):
        row = sparse_matrix.getrow(i)
        on_bits = set(row.nonzero()[1])
        sets.append(on_bits)
    return np.array(sets, dtype=object)

X_sets_converted = sparse_to_sets(X_converted)
clf_converted = LaplacianNB()
clf_converted.fit(X_sets_converted, y)
predictions_converted = clf_converted.predict(X_sets_converted)

print(f"Predictions with converted data: {predictions_converted}")

print("\n✅ All examples completed successfully!")
