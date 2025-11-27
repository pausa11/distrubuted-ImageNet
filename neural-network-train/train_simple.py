"""
Simple training script using raw Tiny-ImageNet data (no WebDataset).
This is for validating the model and training process.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
from pathlib import Path
from torchvision.models import resnet50


def get_device():
    """Auto-detect the best available device."""
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    # Check for CUDA
    elif torch.cuda.is_available():
        return torch.device("cuda")
    # Fallback to CPU
    else:
        return torch.device("cpu")


def get_model(num_classes=200, pretrained=False):
    """Create ResNet50 model for Tiny-ImageNet."""
    model = resnet50(weights=None)
    # Modify final layer for num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_data_loaders(data_dir, batch_size=64, num_workers=4):
    """
    Create data loaders for Tiny-ImageNet using ImageFolder.
    
    Args:
        data_dir: Path to tiny-imagenet-200 directory
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader
    """
    
    # Training transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Training dataset
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation dataset
    val_dir = os.path.join(data_dir, 'val')
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Print progress
        if batch_idx % 50 == 0:
            batch_loss = loss.item()
            batch_acc = 100. * correct / total
            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {batch_loss:.4f} Acc: {batch_acc:.2f}%')
    
    epoch_time = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    avg_acc = 100. * correct / total
    
    print(f'\nEpoch {epoch} Training Summary:')
    print(f'  Average Loss: {avg_loss:.4f}')
    print(f'  Average Accuracy: {avg_acc:.2f}%')
    print(f'  Time: {epoch_time:.2f}s')
    
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Statistics
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Print progress
            if batch_idx % 50 == 0:
                batch_loss = loss.item()
                batch_acc = 100. * correct / total
                print(f'Validation [{batch_idx}/{len(val_loader)}] '
                      f'Loss: {batch_loss:.4f} Acc: {batch_acc:.2f}%')
    
    val_time = time.time() - start_time
    avg_loss = running_loss / len(val_loader)
    avg_acc = 100. * correct / total
    
    print(f'\nEpoch {epoch} Validation Summary:')
    print(f'  Average Loss: {avg_loss:.4f}')
    print(f'  Average Accuracy: {avg_acc:.2f}%')
    print(f'  Time: {val_time:.2f}s')
    
    return avg_loss, avg_acc


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Tiny-ImageNet training')
    parser.add_argument('--data_dir', type=str, default='data/tiny-imagenet-200',
                        help='Path to tiny-imagenet-200 directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_simple',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\nCreating model...")
    model = get_model(num_classes=200).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        
        # Save last checkpoint
        last_path = os.path.join(args.checkpoint_dir, 'last.pth')
        torch.save(checkpoint, last_path)
        print(f"\nSaved checkpoint to {last_path}")
        
        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"New best validation accuracy: {val_acc:.2f}%")
            print(f"Saved best checkpoint to {best_path}")
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Complete")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        print(f"{'='*60}\n")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
