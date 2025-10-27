import torch
import torch.nn as nn
import torchvision.models as models

def resnet50_tiny(num_classes=200):
    m = models.resnet50(weights=None)
    # Tiny-ImageNet tiene 200 clases
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m
