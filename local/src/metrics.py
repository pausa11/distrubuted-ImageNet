import torch
from torch.utils.data import DataLoader

@torch.no_grad()
def eval_top1_acc(model, loader: DataLoader, device="cuda"):
    model.eval()
    correct, total = 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return correct / max(1,total)
