import os
import argparse
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

if __name__ == "__main__":
    mp.set_start_method('fork', force=True)

import hivemind
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from typing import Tuple, List, Dict, Optional

class TinyImageNetValDataset(Dataset):
    """
    Val dataset para el formato original de Tiny-ImageNet-200:
      root/
        val/
          images/
          val_annotations.txt   (archivo con: <img> <wnid> <x0> <y0> <x1> <y1>)
      Además, en root/ existe wnids.txt con el listado de las 200 clases (wnid por línea).
    """
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        # wnids -> idx
        wnids_path = os.path.join(root, "wnids.txt")
        with open(wnids_path, "r") as f:
            wnids = [line.strip() for line in f if line.strip()]
        self.wnid_to_idx: Dict[str, int] = {wnid: i for i, wnid in enumerate(sorted(wnids))}

        images_dir = os.path.join(root, "val", "images")
        annot_path = os.path.join(root, "val", "val_annotations.txt")

        self.samples: List[Tuple[str, int]] = []
        with open(annot_path, "r") as f:
            for line in f:
                # formato: filename \t wnid \t x0 \t y0 \t x1 \t y1
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    filename, wnid = parts[0], parts[1]
                    img_path = os.path.join(images_dir, filename)
                    if os.path.isfile(img_path) and wnid in self.wnid_to_idx:
                        self.samples.append((img_path, self.wnid_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

def build_model(num_classes: int = 200, pretrained: bool = True) -> nn.Module:
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None)
    # Cambiar la capa final a 200 clases
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def get_dataloaders_tiny_imagenet(
    data_root: str,
    batch: int,
    workers: int,
    device: torch.device
):
    """
    Crea loaders de Tiny-ImageNet-200.
    - Si val está reordenado en subcarpetas por clase: usa ImageFolder.
    - Si está en formato original (val/images + val_annotations.txt): usa TinyImageNetValDataset.
    """
    # ResNet50 espera 224x224 y normalización de ImageNet
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))

    train_t = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # TRAIN siempre es ImageFolder: tiny-imagenet-200/train/<class>/images...
    train_dir = os.path.join(data_root, "train")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_t)

    # VAL: detectar si está en subcarpetas por clase
    val_dir = os.path.join(data_root, "val")
    val_subdirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    if "images" in val_subdirs:
        # Formato original → dataset especial
        val_dataset = TinyImageNetValDataset(root=data_root, transform=val_t)
    else:
        # Ya está reordenado en subcarpetas por clase → ImageFolder
        val_dataset = datasets.ImageFolder(val_dir, transform=val_t)

    # pin_memory solo acelera CPU→CUDA, no MPS
    pin = (device.type == "cuda")
    persistent = workers > 0

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=max(256, batch),
        num_workers=min(2, workers),
        pin_memory=pin,
        persistent_workers=(min(2, workers) > 0),
        prefetch_factor=2 if min(2, workers) > 0 else None,
        drop_last=False,
    )

    return train_loader, val_loader

def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model_was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            # Para estabilidad: non_blocking solo CUDA; contiguidad en MPS
            nb = (device.type == "cuda")
            xb = xb.to(device, non_blocking=nb)
            yb = yb.to(device, non_blocking=nb)
            if device.type == "mps":
                xb = xb.contiguous()

            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    accuracy = 100.0 * correct / max(1, total)
    if model_was_training:
        model.train()
    return accuracy

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, out_dir: str, epoch_idx: int, acc: float):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "best_checkpoint.pt")
    torch.save({
        "epoch": epoch_idx,
        "val_accuracy": acc,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
    }, path)
    return path

def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    if not os.path.exists(path):
        return None, -1.0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["opt_state"])

    epoch = checkpoint.get("epoch", 0)
    acc = checkpoint.get("val_accuracy", -1.0)

    print(f"✓ Checkpoint cargado desde: {path}")
    print(f"  Época: {epoch}, Accuracy: {acc:.2f}%")

    return epoch, acc

def parse_arguments():
    p = argparse.ArgumentParser(description="Tiny-ImageNet-200 x ResNet50 x Hivemind")
    p.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None,
                   help="Forzar dispositivo (cpu, cuda o mps). Por defecto: auto-detectar")
    p.add_argument("--use_checkpoint", action="store_true",
                   help="Cargar checkpoint si existe en ./checkpoints/best_checkpoint.pt")
    p.add_argument("--initial_peer", type=str, default=None,
                   help="Multiaddr de un peer inicial para bootstrap (opcional en el primer nodo)")
    p.add_argument("--val_every", type=int, default=5,
                   help="Frecuencia de validación (épocas globales). Ej: 5 = validar cada 5.")
    return p.parse_args()

def select_device(cli_device: Optional[str]) -> torch.device:
    if cli_device:
        return torch.device(cli_device)

    # Preferir MPS (Apple Silicon) si está disponible
    mps_available = (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    if mps_available:
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

def main():
    args = parse_arguments()

    DATA_ROOT = "./data/tiny-imagenet-200"
    RUN_ID = "tiny_imagenet_resnet50"
    BATCH = 32
    TARGET_GLOBAL_BSZ = 30_000
    EPOCHS = 2
    LR = 1e-3
    MOMENTUM = 0.9
    WORKERS = 2
    MATCHMAKING_TIME = 1.5
    AVERAGING_TIMEOUT = 6.0
    CHECKPOINT_DIR = "./checkpoints"

    # Device
    device = select_device(args.device)
    print(f"\nDevice: {device}")
    if device.type == "mps":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # DataLoaders (Tiny-ImageNet-200)
    train_loader, val_loader = get_dataloaders_tiny_imagenet(DATA_ROOT, BATCH, WORKERS, device)

    # Modelo ResNet50 (200 clases)
    model = build_model(num_classes=200, pretrained=True)

    # channels_last SOLO en CUDA (no en MPS por bug en backward)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    model = model.to(device)

    # Optimizador base
    base_optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    # DHT
    dht_kwargs = dict(
        host_maddrs=["/ip4/0.0.0.0/tcp/0"],
        start=True
    )
    if args.initial_peer:
        dht_kwargs["initial_peers"] = [args.initial_peer]
    dht = hivemind.DHT(**dht_kwargs)

    maddrs = [str(m) for m in dht.get_visible_maddrs()]
    print("\n=== Hivemind DHT ===")
    for m in maddrs:
        print("VISIBLE_MADDR:", m)
    if not args.initial_peer:
        print("\n⚠️  Usa una dirección que NO sea 127.0.0.1 como --initial_peer")
        print("    Busca la que tenga tu IP local (192.168.x.x o 10.0.x.x)")

    # Checkpoint (opcional)
    best_accuracy = -1.0
    start_epoch = 0
    if args.use_checkpoint:
        ckpt_path = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt")
        start_epoch, best_accuracy = load_checkpoint(ckpt_path, model, base_optimizer, device)

    # Optimizer de Hivemind
    opt = hivemind.Optimizer(
        dht=dht,
        run_id=RUN_ID,
        batch_size_per_step=BATCH,
        target_batch_size=TARGET_GLOBAL_BSZ,
        optimizer=base_optimizer,
        use_local_updates=True,
        matchmaking_time=MATCHMAKING_TIME,
        averaging_timeout=AVERAGING_TIMEOUT,
        verbose=True,
    )

    target_epochs = EPOCHS
    last_seen_epoch = getattr(opt, "local_epoch", 0)
    checkpoint_path = None

    print(f"\nEntrenando hasta {target_epochs} épocas globales (target_batch_size={TARGET_GLOBAL_BSZ}).")
    if args.use_checkpoint and best_accuracy > 0:
        print(f"Continuando desde mejor accuracy: {best_accuracy:.2f}%")

    try:
        with tqdm(total=None) as pbar:
            while True:
                for xb, yb in train_loader:
                    # Para estabilidad: non_blocking solo CUDA; contiguidad en MPS
                    nb = (device.type == "cuda")

                    xb = xb.to(device, non_blocking=nb)
                    yb = yb.to(device, non_blocking=nb)

                    if device.type == "cuda":
                        xb = xb.to(memory_format=torch.channels_last)
                    elif device.type == "mps":
                        xb = xb.contiguous()

                    opt.zero_grad()
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb)
                    loss.backward()
                    opt.step()

                    pbar.set_description(
                        f"loss={loss.item():.4f}  epoch_g={getattr(opt,'local_epoch',0)}  best={best_accuracy:.2f}%"
                    )
                    pbar.update()

                    current_epoch = getattr(opt, "local_epoch", last_seen_epoch)
                    if current_epoch != last_seen_epoch:
                        do_eval = (current_epoch == 1) or (current_epoch % args.val_every == 0) or (current_epoch >= target_epochs)
                        if do_eval:
                            val_acc = evaluate_accuracy(model, val_loader, device)
                            tqdm.write(f"[Época {current_epoch}] Accuracy validación: {val_acc:.2f}%")

                            if val_acc > best_accuracy:
                                ckpt_path = save_checkpoint(model, base_optimizer, CHECKPOINT_DIR, current_epoch, val_acc)
                                best_accuracy = val_acc
                                checkpoint_path = ckpt_path
                                tqdm.write(f"↑ Nuevo mejor accuracy ({best_accuracy:.2f}%). Checkpoint guardado: {ckpt_path}")
                            else:
                                tqdm.write(f"↔ No mejora (best={best_accuracy:.2f}%). No se guarda checkpoint.")

                        last_seen_epoch = current_epoch

                        if current_epoch >= target_epochs:
                            tqdm.write(f"✓ Alcanzadas {current_epoch} épocas globales. Terminando...")
                            raise StopIteration
    except StopIteration:
        pass

    if checkpoint_path or best_accuracy > 0:
        print(f"\nEntrenamiento finalizado. Mejor accuracy: {best_accuracy:.2f}%")
        if checkpoint_path:
            print(f"Mejor checkpoint: {checkpoint_path}")
    else:
        print("\nEntrenamiento finalizado. No se guardaron checkpoints (no hubo mejora).")

if __name__ == "__main__":
    main()
