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
import json
import hashlib

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
    def __init__(self, bucket_name: str, prefix: str, transform=None, public_access: bool = False):
        """
        A Dataset that streams images from a GCS bucket.
        Expects structure: prefix/class_name/image.jpg
        """
        if not GCS_AVAILABLE:
            raise ImportError("google-cloud-storage is required for GCSImageFolder. Run `pip install google-cloud-storage`.")
        
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/')
        self.transform = transform
        self.public_access = public_access
        
        # Cache logic
        cache_key = hashlib.md5(f"{bucket_name}_{self.prefix}".encode()).hexdigest()
        cache_file = f"gcs_cache_{cache_key}.json"
        
        loaded_from_cache = False
        if os.path.exists(cache_file):
            print(f"Found GCS cache: {cache_file}. Loading...")
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.classes = data['classes']
                    self.samples = data['samples'] # list of [blob_name, class_name]
                    loaded_from_cache = True
                    print(f"Loaded {len(self.samples)} images from cache.")
            except Exception as e:
                print(f"Failed to load cache: {e}. Will re-list blobs.")
        
        # Do NOT initialize client here to avoid fork-safety issues with multiprocessing
        self.client = None
        self.bucket = None

        if not loaded_from_cache:
            
            # We need a temporary client just for listing blobs during init (main process)
            if self.public_access:
                tmp_client = storage.Client.create_anonymous_client()
            else:
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
            
            # Save to cache
            print(f"Saving GCS listing to cache: {cache_file}")
            with open(cache_file, 'w') as f:
                json.dump({
                    'classes': self.classes,
                    'samples': self.samples
                }, f)
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
        # Lazy initialization of GCS client (per worker process)
        if self.client is None:
            if self.public_access:
                self.client = storage.Client.create_anonymous_client()
            else:
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
    val_prefix: Optional[str] = None,
    gcs_public: bool = False
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
        
        train_dataset = GCSImageFolder(bucket_name, final_train_prefix, transform=train_t, public_access=gcs_public)
        val_dataset   = GCSImageFolder(bucket_name, final_val_prefix,   transform=val_t, public_access=gcs_public)
        
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
import itertools

def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> float:
    model_was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # Limit validation to max_batches if specified
        if max_batches is not None:
             loader = itertools.islice(loader, max_batches)
             total_batches = max_batches
        else:
             total_batches = len(loader)

        for xb, yb in tqdm(loader, total=total_batches, desc="Validating", leave=False):
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


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, out_dir: str, epoch_idx: int, acc: float, filename: str = "best_checkpoint.pt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
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
                   help="Cargar checkpoint si existe (latest o best)")
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
    p.add_argument("--gcs_public", action="store_true",
                   help="Usar acceso anónimo para bucket público (sin credenciales)")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size local (por defecto: 64). Reducir si hay OOM.")
    p.add_argument("--target_batch_size", type=int, default=30000,
                   help="Batch size global objetivo para Hivemind (por defecto: 30000).")
    p.add_argument("--val_batches", type=int, default=100,
                   help="Número máximo de batches para validar (default: 100). None para validar todo.")
    p.add_argument("--no_initial_val", action="store_true",
                   help="No forzar validación en la época 1.")
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
    BATCH = args.batch_size
    TARGET_GLOBAL_BSZ = args.target_batch_size
    EPOCHS = 2
    LR = 1e-3
    MOMENTUM = 0.9
    WORKERS = 0          # En MPS con fork, 0 es OBLIGATORIO para evitar SegFaults.
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
        val_prefix=args.val_prefix,
        gcs_public=args.gcs_public
    )

    # Modelo ResNet50 (1000 clases)
    # MASTER en CPU (para Hivemind)
    model = build_model(num_classes=1000, pretrained=False)
    model = model.to("cpu")

    # SHADOW en Device (para cómputo MPS/GPU)
    model_on_device = build_model(num_classes=1000, pretrained=False)
    model_on_device = model_on_device.to(device)

    # channels_last SOLO en CUDA (no en MPS por bug en backward)
    if device.type == "cuda":
        model_on_device = model_on_device.to(memory_format=torch.channels_last)

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

    # --- LOAD CHECKPOINT AFTER OPT CREATION ---
    if args.use_checkpoint:
        # Intentar cargar latest primero
        latest_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
        best_path = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt")
        
        if os.path.exists(latest_path):
            print(f"Intentando reanudar desde LATEST: {latest_path}")
            # IMPORTANTE: Pasamos 'opt' (Hivemind Optimizer) en lugar de 'base_optimizer'
            start_epoch, acc = load_checkpoint(latest_path, model, opt, device)
            
            if os.path.exists(best_path):
                checkpoint = torch.load(best_path, map_location='cpu')
                best_accuracy = checkpoint.get("val_accuracy", acc)
            else:
                best_accuracy = acc
        elif os.path.exists(best_path):
            print(f"Intentando reanudar desde BEST: {best_path}")
            start_epoch, best_accuracy = load_checkpoint(best_path, model, opt, device)
        else:
            print("No se encontraron checkpoints para cargar.")
    # ------------------------------------------

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

                    # 1. Sincronizar pesos CPU -> Device
                    with torch.no_grad():
                        for p_cpu, p_dev in zip(model.parameters(), model_on_device.parameters()):
                            p_dev.copy_(p_cpu)
                        for b_cpu, b_dev in zip(model.buffers(), model_on_device.buffers()):
                            b_dev.copy_(b_cpu)

                    # 2. Forward/Backward en Device
                    model_on_device.train()
                    model_on_device.zero_grad()
                    
                    logits = model_on_device(xb)
                    loss = F.cross_entropy(logits, yb)
                    loss.backward()

                    # 3. Sincronizar gradientes Device -> CPU
                    with torch.no_grad():
                        for p_cpu, p_dev in zip(model.parameters(), model_on_device.parameters()):
                            if p_dev.grad is not None:
                                if p_cpu.grad is None:
                                    p_cpu.grad = torch.zeros_like(p_cpu)
                                p_cpu.grad.copy_(p_dev.grad)

                    # 4. Step en CPU (Hivemind)
                    opt.step()
                    opt.zero_grad()

                    pbar.set_description(
                        f"loss={loss.item():.4f}  epoch_g={getattr(opt,'local_epoch',0)}  best={best_accuracy:.2f}%"
                    )
                    pbar.update()

                    current_epoch = getattr(opt, "local_epoch", last_seen_epoch)
                    if current_epoch != last_seen_epoch:
                        force_initial = (current_epoch == 1 and not args.no_initial_val)
                        do_eval = force_initial or (current_epoch % args.val_every == 0) or (current_epoch >= target_epochs)
                        if do_eval:
                            tqdm.write(f"Starting validation for epoch {current_epoch} (max_batches={args.val_batches})...")
                            
                            # Sync weights to device before eval to ensure latest state
                            with torch.no_grad():
                                for p_cpu, p_dev in zip(model.parameters(), model_on_device.parameters()):
                                    p_dev.copy_(p_cpu)
                                for b_cpu, b_dev in zip(model.buffers(), model_on_device.buffers()):
                                    b_dev.copy_(b_cpu)

                            val_acc = evaluate_accuracy(model_on_device, val_loader, device, max_batches=args.val_batches)
                            tqdm.write(f"[Época {current_epoch}] Accuracy validación: {val_acc:.2f}%")

                            # SIEMPRE guardar el latest checkpoint (usando opt, no base_optimizer)
                            save_checkpoint(model, opt, CHECKPOINT_DIR, current_epoch, val_acc, filename="latest_checkpoint.pt")
                            
                            if val_acc > best_accuracy:
                                ckpt_path = save_checkpoint(model, opt, CHECKPOINT_DIR, current_epoch, val_acc, filename="best_checkpoint.pt")
                                best_accuracy = val_acc
                                checkpoint_path = ckpt_path
                                tqdm.write(f"↑ Nuevo mejor accuracy ({best_accuracy:.2f}%). Checkpoint guardado: {ckpt_path}")
                            else:
                                tqdm.write(f"↔ No mejora (best={best_accuracy:.2f}%). Guardado latest_checkpoint.pt.")

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
