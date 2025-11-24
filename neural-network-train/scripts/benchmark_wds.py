import time
import torch
import os
import argparse
import sys
import os

# Add src to path to import datasets
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from datasets import get_webdataset_loader

def benchmark(num_workers, batch_size=64, steps=50):
    print(f"\n--- Benchmarking with num_workers={num_workers} ---")
    
    # Force MPS if available, else CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    try:
        loader = get_webdataset_loader(
            bucket_name="caso-estudio-2",
            prefix="imagenet-wds",
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            is_train=True,
            total_shards=641,
            train_prefix="train/train"
        )

        start_time = time.time()
        count = 0
        
        # Warmup
        print("Warming up...")
        iter_loader = iter(loader)
        next(iter_loader)
        
        print(f"Measuring over {steps} steps...")
        t0 = time.time()
        for i in range(steps):
            batch = next(iter_loader)
            count += batch_size
            if i % 10 == 0:
                print(f"Step {i}/{steps}", end='\r')
                
        t1 = time.time()
        duration = t1 - t0
        throughput = count / duration
        
        print(f"\nResult: {throughput:.2f} samples/sec (Total time: {duration:.2f}s)")
        return throughput

    except Exception as e:
        print(f"\nFAILED with num_workers={num_workers}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, nargs='+', default=[0, 2, 4], help="List of worker counts to test")
    args = parser.parse_args()

    results = {}
    for w in args.workers:
        throughput = benchmark(w)
        results[w] = throughput
        
    print("\n=== SUMMARY ===")
    for w, t in results.items():
        print(f"Workers={w}: {t:.2f} samples/sec")
