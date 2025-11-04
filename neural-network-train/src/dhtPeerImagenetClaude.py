import os
import argparse
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from PIL import Image

if __name__ == "__main__":
    mp.set_start_method('fork', force=True)

import hivemind

# =========================
# Utilidades de dispositivo
# =========================
def pick_device(forced: str | None) -> torch.device:
    """
    Retorna torch.device considerando --device y disponibilidad real.
    Prioridad auto: cuda > mps > cpu.
    Si se fuerza un device y no está disponible, cae a la mejor alternativa posible.
    """
    def mps_available() -> bool:
        try:
            return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        except Exception:
            return False

    if forced:
        forced = forced.lower()
        if forced == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if forced == "mps" and mps_available():
            return torch.device("mps")
        if forced == "cpu":
            return torch.device("cpu")
        # Forzado pero no disponible → degradar
        print(f"⚠️  '{forced}' no disponible. Usando mejor alternativa.")
    # Auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================
# Dataset Tiny ImageNet
# =========================
class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Ruta a tiny-imagenet-200/
            split: 'train' o 'val'
            transform: Transformaciones de torchvision
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        if split == 'train':
            self._load_train_data()
        else:
            self._load_val_data()
    
    def _load_train_data(self):
        train_dir = os.path.join(self.root_dir, 'train')
        
        # Crear mapeo de clases
        classes = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Cargar imágenes
        for class_name in classes:
            class_dir = os.path.join(train_dir, class_name, 'images')
            class_idx = self.class_to_idx[class_name]
            
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        print(f"✓ Train: {len(self.samples)} imágenes, {len(self.class_to_idx)} clases")
    
    def _load_val_data(self):
        val_dir = os.path.join(self.root_dir, 'val')
        
        # Primero cargar el mapeo de clases desde train
        train_dir = os.path.join(self.root_dir, 'train')
        classes = sorted([d for d in os.listdir(train_dir) 
                         if os.path.isdir(os.path.join(train_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Leer anotaciones de validación
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        img_to_class = {}
        
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        img_name = parts[0]
                        class_name = parts[1]
                        img_to_class[img_name] = class_name
        
        # Cargar imágenes
        val_images_dir = os.path.join(val_dir, 'images')
        if os.path.exists(val_images_dir):
            for img_name in os.listdir(val_images_dir):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(val_images_dir, img_name)
                    class_name = img_to_class.get(img_name)
                    
                    if class_name and class_name in self.class_to_idx:
                        class_idx = self.class_to_idx[class_name]
                        self.samples.append((img_path, class_idx))
        
        print(f"✓ Val: {len(self.samples)} imágenes, {len(self.class_to_idx)} clases")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            # Retornar una imagen en blanco si falla
            dummy_img = Image.new('RGB', (64, 64), color='black')
            if self.transform:
                dummy_img = self.transform(dummy_img)
            return dummy_img, label


def build_model(num_classes=200):
    """Crea ResNet50 para Tiny ImageNet (200 clases)"""
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    print(f"✓ Modelo: ResNet50 ({num_classes} clases)")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros totales: {total_params:,}")
    print(f"  Parámetros entrenables: {trainable_params:,}")
    return model


def get_dataloaders(data_root: str, batch: int, workers: int):
    """Carga Tiny ImageNet con transformaciones apropiadas"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    trainDataset = TinyImageNetDataset(root_dir=data_root, split='train', transform=train_transform)
    validationDataset = TinyImageNetDataset(root_dir=data_root, split='val', transform=val_transform)
    train_loader = DataLoader(
        trainDataset,
        shuffle=True,
        batch_size=batch,
        num_workers=workers,
        pin_memory=False,   # se activará más abajo solo en CUDA
        drop_last=False,
    )
    validation_loader = DataLoader(
        validationDataset,
        shuffle=False,
        batch_size=max(128, batch),
        num_workers=min(2, workers),
        pin_memory=False,   # se activará más abajo solo en CUDA
        drop_last=False,
    )
    return train_loader, validation_loader


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """Evalúa accuracy (top-1 y top-5)"""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred_top1 = logits.argmax(dim=1)
            correct_top1 += (pred_top1 == yb).sum().item()
            _, pred_top5 = logits.topk(5, dim=1)
            # yb[i] in pred_top5[i] usando operaciones tensoriales
            correct_top5 += (pred_top5.eq(yb.view(-1, 1))).sum().item()
            total += yb.size(0)
    acc_top1 = 100.0 * correct_top1 / max(1, total)
    acc_top5 = 100.0 * correct_top5 / max(1, total)
    model.train()
    return acc_top1, acc_top5


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, out_dir: str, 
                   epoch_idx: int, acc_top1: float, acc_top5: float):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "best_checkpoint.pt")
    torch.save({
        "epoch": epoch_idx,
        "val_accuracy_top1": acc_top1,
        "val_accuracy_top5": acc_top5,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
    }, path)
    return path


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    if not os.path.exists(path):
        return None, -1.0, -1.0
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["opt_state"])
    epoch = checkpoint.get("epoch", 0)
    acc_top1 = checkpoint.get("val_accuracy_top1", -1.0)
    acc_top5 = checkpoint.get("val_accuracy_top5", -1.0)
    print(f"✓ Checkpoint cargado desde: {path}")
    print(f"  Época: {epoch}, Top-1: {acc_top1:.2f}%, Top-5: {acc_top5:.2f}%")
    return epoch, acc_top1, acc_top5


def parse_arguments():
    p = argparse.ArgumentParser(description="Tiny ImageNet x Hivemind - ResNet50")
    p.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None,
                   help="Forzar dispositivo (cpu, cuda o mps). Por defecto: auto-detectar")
    p.add_argument("--use_checkpoint", action="store_true", 
                   help="Cargar checkpoint si existe")
    p.add_argument("--initial_peer", type=str, default=None, 
                   help="Multiaddr de un peer inicial para bootstrap")
    # Parámetros de optimización
    p.add_argument("--target_batch_size", type=int, default=100000, 
                   help="Batch global objetivo (default: 100000 = ~1 época)")
    p.add_argument("--local_batch_size", type=int, default=64, 
                   help="Batch size local (default: 64)")
    p.add_argument("--matchmaking_time", type=float, default=2.0, 
                   help="Tiempo de búsqueda de peers (default: 2.0s)")
    p.add_argument("--averaging_timeout", type=float, default=10.0, 
                   help="Timeout de sincronización (default: 10.0s)")
    p.add_argument("--num_epochs", type=int, default=2, 
                   help="Número de épocas globales (default: 5)")
    p.add_argument("--compression", action="store_true", 
                   help="Usar compresión FP16")
    p.add_argument("--learning_rate", type=float, default=0.1, 
                   help="Learning rate inicial (default: 0.1)")
    return p.parse_args()


def main():
    args = parse_arguments()

    # Configuración
    DATA_ROOT = "./data/tiny-imagenet-200"
    RUN_ID = "tiny_imagenet_resnet50"
    LR = args.learning_rate
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    WORKERS = 4
    CHECKPOINT_DIR = "./checkpoints"

    # Verificar dataset
    if not os.path.exists(DATA_ROOT):
        print(f"❌ ERROR: No se encontró el dataset en {DATA_ROOT}")
        print(f"   Estructura esperada:")
        print(f"   {DATA_ROOT}/")
        print(f"   ├── train/<clase>/images/*.JPEG")
        print(f"   └── val/images/*.JPEG + val_annotations.txt")
        return

    # Parámetros
    BATCH = args.local_batch_size
    TARGET_GLOBAL_BSZ = args.target_batch_size
    EPOCHS = args.num_epochs
    MATCHMAKING_TIME = args.matchmaking_time
    AVERAGING_TIMEOUT = args.averaging_timeout

    print("\n" + "="*70)
    print("CONFIGURACIÓN - TINY IMAGENET + RESNET50")
    print("="*70)
    print(f"Dataset:                  Tiny ImageNet (200 clases)")
    print(f"Modelo:                   ResNet50")
    print(f"Batch local:              {BATCH}")
    print(f"Target batch global:      {TARGET_GLOBAL_BSZ}")
    print(f"Learning rate:            {LR}")
    print(f"Épocas objetivo:          {EPOCHS}")
    print(f"Matchmaking time:         {MATCHMAKING_TIME}s")
    print(f"Averaging timeout:        {AVERAGING_TIMEOUT}s")
    print(f"Compresión FP16:          {'✓ Activada' if args.compression else '✗ Desactivada'}")
    print("="*70 + "\n")

    # Datos
    print("Cargando dataset...")
    train_loader, validation_loader = get_dataloaders(DATA_ROOT, BATCH, WORKERS)
    
    train_size = len(train_loader.dataset)
    syncs_per_epoch = train_size // TARGET_GLOBAL_BSZ
    if syncs_per_epoch == 0:
        syncs_per_epoch = 1
    total_syncs = EPOCHS * syncs_per_epoch
    
    print(f"\nTamaño dataset train:     {train_size:,} imágenes")
    print(f"Sincronizaciones/época:   {syncs_per_epoch}")
    print(f"Total sincronizaciones:   {total_syncs}")
    print("="*70 + "\n")

    # Modelo
    model = build_model(num_classes=200)
    
    # DHT
    dht_kwargs = dict(host_maddrs=["/ip4/0.0.0.0/tcp/0"], start=True)
    if args.initial_peer:
        dht_kwargs["initial_peers"] = [args.initial_peer]
    dht = hivemind.DHT(**dht_kwargs)
    maddrs = [str(m) for m in dht.get_visible_maddrs()]
    print("\n=== Hivemind DHT ===")
    for m in maddrs:
        print("VISIBLE_MADDR:", m)
    if not args.initial_peer:
        print("\n⚠️  Primer nodo iniciado. Los demás nodos deben usar --initial_peer")

    # Device (con soporte MPS)
    device = pick_device(args.device)
    model.to(device)
    print(f"\nDevice: {device}")

    # Optimizer base
    base_optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=LR, 
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # Checkpoint
    best_accuracy_top1 = -1.0
    best_accuracy_top5 = -1.0
    start_epoch = 0
    if args.use_checkpoint:
        ckpt_path = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt")
        start_epoch, best_accuracy_top1, best_accuracy_top5 = load_checkpoint(
            ckpt_path, model, base_optimizer, device
        )

    # Compresión
    compression_config = None
    if args.compression:
        try:
            compression_config = hivemind.Compression(
                compression=hivemind.CompressionType.FLOAT16,
                min_compression_ratio=1.5
            )
            print("✓ Compresión FP16 habilitada")
        except Exception:
            print("⚠️  Compresión no disponible en esta versión de Hivemind")

    # Optimizer Hivemind
    opt_kwargs = dict(
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
    if compression_config:
        opt_kwargs['grad_compression'] = compression_config
    opt = hivemind.Optimizer(**opt_kwargs)

    # pin_memory solo en CUDA
    if device.type == "cuda":
        train_loader.pin_memory = True
        validation_loader.pin_memory = True

    target_epochs = EPOCHS
    last_seen_epoch = getattr(opt, "local_epoch", 0)
    checkpoint_path = None

    print(f"\nEntrenando hasta {target_epochs} épocas globales.")
    if args.use_checkpoint and best_accuracy_top1 > 0:
        print(f"Continuando desde Top-1: {best_accuracy_top1:.2f}%, Top-5: {best_accuracy_top5:.2f}%")

    # Loop de entrenamiento
    sync_count = 0
    import time
    start_time = time.time()

    try:
        with tqdm(total=None) as pbar:
            while True:
                for xb, yb in train_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    opt.zero_grad()
                    loss = F.cross_entropy(model(xb), yb)
                    loss.backward()
                    
                    step_start = time.time()
                    opt.step()
                    step_time = time.time() - step_start

                    pbar.set_description(
                        f"loss={loss.item():.4f}  epoch={getattr(opt,'local_epoch',0)}  "
                        f"best_top1={best_accuracy_top1:.2f}%  step={step_time:.2f}s"
                    )
                    pbar.update()

                    # Transición de época global
                    current_epoch = getattr(opt, "local_epoch", last_seen_epoch)
                    if current_epoch != last_seen_epoch:
                        sync_count += 1
                        elapsed = time.time() - start_time
                        
                        # Validación
                        val_acc_top1, val_acc_top5 = evaluate_accuracy(model, validation_loader, device)
                        tqdm.write(
                            f"[Época {current_epoch}] Top-1: {val_acc_top1:.2f}%  Top-5: {val_acc_top5:.2f}%  "
                            f"Syncs: {sync_count}  Tiempo: {elapsed/60:.1f}min"
                        )

                        # Guardar si mejora top-1
                        if val_acc_top1 > best_accuracy_top1:
                            ckpt_path = save_checkpoint(
                                model, base_optimizer, CHECKPOINT_DIR, 
                                current_epoch, val_acc_top1, val_acc_top5
                            )
                            best_accuracy_top1 = val_acc_top1
                            best_accuracy_top5 = val_acc_top5
                            checkpoint_path = ckpt_path
                            tqdm.write(f"↑ Nuevo mejor Top-1 ({best_accuracy_top1:.2f}%). Checkpoint guardado.")
                        
                        last_seen_epoch = current_epoch

                        # Parada
                        if current_epoch >= target_epochs:
                            tqdm.write(f"✓ Alcanzadas {current_epoch} épocas. Terminando...")
                            raise StopIteration
    except StopIteration:
        pass

    # Reporte final
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("RESULTADOS FINALES")
    print("="*70)
    print(f"Mejor Top-1 accuracy:     {best_accuracy_top1:.2f}%")
    print(f"Mejor Top-5 accuracy:     {best_accuracy_top5:.2f}%")
    print(f"Tiempo total:             {total_time/60:.1f} min ({total_time:.0f}s)")
    print(f"Total sincronizaciones:   {sync_count}")
    if sync_count > 0:
        print(f"Tiempo por sync:          {total_time/sync_count:.1f}s")
    if checkpoint_path:
        print(f"Mejor checkpoint:         {checkpoint_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
