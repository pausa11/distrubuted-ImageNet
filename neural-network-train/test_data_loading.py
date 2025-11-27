#!/usr/bin/env python3
"""
Quick test to verify ImageFolder data loading works correctly.
"""
import sys
sys.path.insert(0, 'src')

from dhtPeerImagenet import get_data_loaders

print("Testing ImageFolder data loading...")
print("-" * 60)

try:
    train_loader, val_loader = get_data_loaders(
        'data/tiny-imagenet-200',
        batch_size=32,
        num_workers=0
    )
    
    print("\n✅ Data loaders created successfully!")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test loading one batch
    print("\nTesting batch loading...")
    batch_x, batch_y = next(iter(train_loader))
    print(f"Batch images shape: {batch_x.shape}")
    print(f"Batch labels shape: {batch_y.shape}")
    print(f"Label range: [{batch_y.min()}, {batch_y.max()}]")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
