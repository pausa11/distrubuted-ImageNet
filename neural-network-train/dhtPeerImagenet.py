import os
import argparse
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import io

# Try to import google-cloud-storage
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

if __name__ == "__main__":
    mp.set_start_method('fork', force=True)

import hivemind
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from typing import Optional

# =========================
#  GCS Dataset
# =========================
class GCSImageFolder(Dataset):
    def __init__(self, bucket_name: str, prefix: str, transform=None):
        """
        A Dataset that streams images from a GCS bucket.
        Expects structure: prefix/class_name/image.jpg
        """
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage is required for GCSImageFolder. Run `pip install google-cloud-storage`.")
        
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/')
        self.transform = transform
        
        # Do NOT initialize client here to avoid fork-safety issues with multiprocessing
        self.client = None
        self.bucket = None
        
        # We need a temporary client just for listing blobs during init (main process)
        tmp_client = storage.Client()
        tmp_bucket = tmp_client.bucket(bucket_name)
        
        print(f"Listing blobs in gs://{bucket_name}/{self.prefix} ... (this may take a while)")
        blobs = list(tmp_bucket.list_blobs(prefix=self.prefix))
        
        self.samples = []
        self.classes = set()
        
        # Filter for images and parse classes
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for blob in tqdm(blobs, desc="Indexing GCS files"):
            if blob.name.endswith('/'): continue # Skip directories
            ext = os.path.splitext(blob.name)[1].lower()
            if ext in valid_extensions:
                # Structure: prefix/class_name/filename
                # rel_path: class_name/filename
                rel_path = blob.name[len(self.prefix)+1:]
                parts = rel_path.split('/')
                if len(parts) >= 2:
                    class_name = parts[0]
                    self.classes.add(class_name)
                    self.samples.append((blob.name, class_name))
        
        self.classes = sorted(list(self.classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Found {len(self.samples)} images belonging to {len(self.classes)} classes.")
        if len(self.samples) == 0:
            msg = (f"⚠️  WARNING: No images found in gs://{bucket_name}/{self.prefix}\n"
                   f"    Check if the prefix is correct. The script expects: prefix/class_name/image.jpg")
            print(msg)
            raise RuntimeError(msg)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        blob_name, class_name = self.samples[idx]
        label = self.class_to_idx[class_name]
        
        # Lazy initialization of GCS client (per worker process)
        if self.client is None:
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)
        
        # Download image bytes
        blob = self.bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()
        
        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

# =========================
#  ImageNet-1K Dataloaders
# =========================
def get_dataloaders_imagenet(
    data_root: str,
    batch: int,
    workers: int,
    device: torch.device,
    bucket_name: Optional[str] = None,
    train_prefix: Optional[str] = None,
    val_prefix: Optional[str] = None
):
    """
    Espera un árbol:
      data_root/
        train/0..999/*.JPEG
        val/0..999/*.JPEG
    
    O si bucket_name está definido:
      gs://bucket_name/train_prefix/...
      gs://bucket_name/val_prefix/...
    """
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

    if bucket_name:
        print(f"Using GCS Bucket: {bucket_name}")
        
        # Determine prefixes
        # If explicit prefix is given, use it.
        # Otherwise, fall back to data_root/train (or just "train" if data_root is empty/None)
        
        if train_prefix:
            final_train_prefix = train_prefix
        else:
            final_train_prefix = os.path.join(data_root, "train") if data_root else "train"
            
        if val_prefix:
            final_val_prefix = val_prefix
        else:
            final_val_prefix = os.path.join(data_root, "val") if data_root else "val"
            
        print(f"  Train prefix: {final_train_prefix}")
        print(f"  Val prefix:   {final_val_prefix}")
        
        train_dataset = GCSImageFolder(bucket_name, final_train_prefix, transform=train_t)
        val_dataset   = GCSImageFolder(bucket_name, final_val_prefix,   transform=val_t)
        
    else:
        train_dir = os.path.join(data_root, "train")
        val_dir   = os.path.join(data_root, "val")

        train_dataset = datasets.ImageFolder(train_dir, transform=train_t)
        val_dataset   = datasets.ImageFolder(val_dir,   transform=val_t)

    # pin_memory solo ayuda CPU->CUDA; en MPS/CPU no aporta
    pin = (device.type == "cuda")
    # persistent_workers=True can cause deadlocks/hangs on macOS with GCS/multiprocessing
    # We disable it to ensure workers are fresh each epoch.
    persistent = False 
    prefetch_train = 2 if workers > 0 else None

    val_workers = min(4, workers if workers else 0)
    prefetch_val = 2 if val_workers > 0 else None
    persistent_val = False

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=prefetch_train,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=min(128, batch),
        num_workers=val_workers,
        pin_memory=pin,
        persistent_workers=persistent_val,
        prefetch_factor=prefetch_val,
        drop_last=False,
    )
    return train_loader, val_loader


# =========================
#  Modelo (1000 clases)
# =========================
def build_model(num_classes: int = 1000, pretrained: bool = True) -> nn.Module:
    if pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
    else:
        model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# =========================
#  Eval, checkpoints, args
# =========================
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model_was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            nb = (device.type == "cuda")  # non_blocking solo CUDA
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
    p = argparse.ArgumentParser(description="ImageNet-1K x ResNet50 x Hivemind")
    p.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None,
                   help="Forzar dispositivo (cpu, cuda o mps). Por defecto: auto-detectar")
    p.add_argument("--use_checkpoint", action="store_true",
                   help="Cargar checkpoint si existe en ./checkpoints/best_checkpoint.pt")
    p.add_argument("--initial_peer", type=str, default=None,
                   help="Multiaddr de un peer inicial para bootstrap (opcional en el primer nodo)")
    p.add_argument("--val_every", type=int, default=5,
                   help="Frecuencia de validación (épocas globales). Ej: 5 = validar cada 5.")
    p.add_argument("--bucket_name", type=str, default=None,
                   help="Nombre del bucket de GCS para cargar datos (opcional)")
    p.add_argument("--data_root", type=str, default=None,
                   help="Ruta base de los datos (local o prefijo en bucket). Si no se especifica, usa ~/data/imagenet-1k")
    p.add_argument("--train_prefix", type=str, default=None,
                   help="Prefijo específico para datos de entrenamiento en GCS (ej: ILSVRC2012_img_train)")
    p.add_argument("--val_prefix", type=str, default=None,
                   help="Prefijo específico para datos de validación en GCS (ej: ILSVRC2012_img_val)")
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


# =========================
#  Main
# =========================
def main():
    args = parse_arguments()

    # IMPORTANTE: apunta al target_dir que generaste con el unpack:
    # ~/data/imagenet-1k/{train,val}/0..999
    if args.data_root:
        DATA_ROOT = args.data_root
    else:
        DATA_ROOT = os.path.expanduser("~/data/imagenet-1k")
        
    RUN_ID = "imagenet1k_resnet50"
    BATCH = 64
    TARGET_GLOBAL_BSZ = 30_000
    EPOCHS = 2
    LR = 1e-3
    MOMENTUM = 0.9
    WORKERS = 4          # En MPS, si notas inestabilidad, prueba WORKERS=0
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

    # DataLoaders (ImageNet-1K)
    train_loader, val_loader = get_dataloaders_imagenet(
        DATA_ROOT, BATCH, WORKERS, device, 
        bucket_name=args.bucket_name,
        train_prefix=args.train_prefix,
        val_prefix=args.val_prefix
    )

    # Modelo ResNet50 (1000 clases)
    model = build_model(num_classes=1000, pretrained=True)

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
        # Si usas GCS, puede que quieras aumentar el timeout si la carga de datos es lenta
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
                    nb = (device.type == "cuda")  # non_blocking solo CUDA

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
