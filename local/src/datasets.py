import os
from torchvision import transforms, datasets

def get_tiny_imagenet(root_dir: str):
    """
    Espera estructura estándar:
    root_dir/
      ├─ train/ (subcarpetas por clase con images)
      └─ val/   (subcarpeta 'images' y un val_annotations.txt -> usa ImageFolder corregida si es necesario)
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(root_dir, "train")
    val_dir   = os.path.join(root_dir, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tfms)

    return train_ds, val_ds
