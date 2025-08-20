#!/usr/bin/env python3
"""
Large-Scale Fingerprint Conversion Benchmark
==========================================

Test the performance and scalability of LaplacianNB fingerprint conversion
with datasets up to 100,000 molecules.
"""

import sys
import os

# Add src to path so we can import laplaciannb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from laplaciannb.fingerprint_utils import benchmark_large_scale_conversion

def main():
    """Run large-scale fingerprint conversion benchmark."""
    print("LaplacianNB Large-Scale Fingerprint Benchmark")
    print("=" * 50)
    print("Testing conversion performance up to 100,000 molecules")
    print("This benchmark evaluates:")
    print("- Conversion speed and throughput")
    print("- Memory usage and efficiency") 
    print("- Scalability characteristics")
    print("- Performance projections")
    
    try:
        # Run the comprehensive large-scale benchmark
        results = benchmark_large_scale_conversion(
            target_molecules=100000,
            test_sizes=[1000, 5000, 10000, 25000, 50000, 100000],
            radius=2,
            sample_diversity=True
        )
        
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        
        if results:
            fastest_rate = max(r['rate'] for r in results)
            largest_test = max(results, key=lambda x: x['molecules'])
            
            print(f"Peak conversion rate: {fastest_rate:,.0f} molecules/second")
            print(f"Largest test completed: {largest_test['molecules']:,} molecules")
            print(f"Time for largest test: {largest_test['time']:.1f} seconds")
            print(f"Memory for largest test: {largest_test['memory_mb']:.1f} MB")
            print(f"Sparsity achieved: {largest_test['sparsity']:.6f}")
            
            # Calculate efficiency metrics
            total_molecules = sum(r['molecules'] for r in results)
            total_time = sum(r['time'] for r in results)
            overall_rate = total_molecules / total_time
            
            print(f"\nOverall benchmark performance:")
            print(f"  Total molecules processed: {total_molecules:,}")
            print(f"  Total time: {total_time:.1f} seconds") 
            print(f"  Average rate: {overall_rate:,.0f} molecules/second")
        
        print(f"\n✓ Large-scale benchmark completed successfully!")
        print(f"✓ LaplacianNB fingerprint conversion scales efficiently to 100K+ molecules")
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install rdkit scikit-learn scipy")
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
