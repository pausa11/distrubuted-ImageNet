import os
import argparse
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import hivemind
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Optional
import itertools
import warnings
import socket
import glob
from metrics import ResourceMonitor, TrainingLogger

warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32 behavior.*")

if __name__ == "__main__":
    mp.set_start_method('fork', force=True)

def build_model(num_classes: int = 200) -> nn.Module:
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def get_data_loaders(data_dir, batch_size=64, num_workers=0):
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

def evaluate_accuracy(model: nn.Module, loader, device: torch.device, max_batches: Optional[int] = None) -> tuple:
    """
    Evaluate model on validation data.
    
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model_was_training = model.training
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        # Limit validation to max_batches if specified
        if max_batches is not None:
             loader_iter = itertools.islice(loader, max_batches)
             total_batches = max_batches
        else:
             loader_iter = loader
             try:
                 total_batches = len(loader)
             except TypeError:
                 # Fallback for IterableDataset which might not have len() on loader
                 # but we can try dataset len / batch_size
                 if hasattr(loader, 'dataset') and hasattr(loader.dataset, '__len__') and hasattr(loader, 'batch_size'):
                     total_batches = len(loader.dataset) // loader.batch_size
                 else:
                     total_batches = None

        for xb, yb in tqdm(loader_iter, total=total_batches, desc="Validating", leave=False):
            nb = (device.type == "cuda")  # non_blocking solo CUDA
            xb = xb.to(device, non_blocking=nb)
            yb = yb.to(device, non_blocking=nb)
            if device.type == "mps":
                xb = xb.contiguous()

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            
            running_loss += loss.item()
            num_batches += 1
            
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    
    avg_loss = running_loss / max(1, num_batches)
    accuracy = 100.0 * correct / max(1, total)
    
    if model_was_training:
        model.train()
    
    return avg_loss, accuracy

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, out_dir: str, epoch_idx: int, acc: float, filename: str = "best_checkpoint.pt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save({
        "epoch": epoch_idx,
        "val_accuracy": acc,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
    }, path)
    return path

def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, device: torch.device):
    if not os.path.exists(path):
        return None, -1.0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["opt_state"])
    
    if scheduler and "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    epoch = checkpoint.get("epoch", 0)
    acc = checkpoint.get("val_accuracy", -1.0)

    print(f"✓ Checkpoint cargado desde: {path}")
    print(f"  Época: {epoch}, Accuracy: {acc:.2f}%")

    return epoch, acc

def parse_arguments():
    p = argparse.ArgumentParser(description="Tiny-ImageNet x ResNet50 x Hivemind")
    p.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default=None,
                   help="Forzar dispositivo (cpu, cuda o mps). Por defecto: auto-detectar")

    p.add_argument("--initial_peer", type=str, default=None,
                   help="Multiaddr de un peer inicial para bootstrap (opcional en el primer nodo)")
    p.add_argument("--val_every", type=int, default=1,
                   help="Frecuencia de validación (épocas globales). Ej: 5 = validar cada 5.")
    
    # Data loading
    p.add_argument("--data_dir", type=str, default="data/tiny-imagenet-200",
                   help="Path to tiny-imagenet-200 directory (default: data/tiny-imagenet-200)")
    p.add_argument("--num_workers", type=int, default=0,
                   help="Number of data loading workers (default: 0 for MPS compatibility)")

    p.add_argument("--batch_size", type=int, default=64,
                   help="Batch size local (por defecto: 64). Reducir si hay OOM.")
    p.add_argument("--target_batch_size", type=int, default=100000,
                   help="Batch size global objetivo para Hivemind (por defecto: 16384).")
    p.add_argument("--val_batches", type=int, default=None,
                   help="Número máximo de batches para validar (default: 100). None para validar todo.")
    p.add_argument("--no_initial_val", action="store_true",
                   help="No forzar validación en la época 1.")
    p.add_argument("--epochs", type=int, default=2000,
                   help="Número total de épocas globales (default: 2000).")
    p.add_argument("--host_port", type=int, default=31337,
                   help="Puerto TCP para escuchar conexiones entrantes (default: 31337).")
                   
    # Hyperparameters
    p.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.05)")
    p.add_argument("--scheduler_milestones", type=int, nargs='+', default=[1000, 1600, 1800], 
                   help="Milestones for MultiStepLR scheduler (default: 1000 1600 1800)")
    p.add_argument("--scheduler_gamma", type=float, default=0.1, help="Gamma for scheduler (default: 0.1)")

    # Automated Peer Discovery
    p.add_argument("--announce_gcs_path", type=str, default=None,
                   help="GCS path to write this peer's address to (e.g. gs://bucket/peer.txt). For the Initial Peer.")
    p.add_argument("--fetch_gcs_path", type=str, default=None,
                   help="GCS path to read the initial peer address from. For Worker Peers.")

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
        # Configurar TF32 para Ampere+ (solo si hay CUDA)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")

    return torch.device("cpu")

def get_next_log_paths(base_dir="stats"):
    """
    Generates unique paths for system and training metrics based on the hostname
    and an incremental run number.
    
    Structure: ./stats/<hostname>/run_<N>_system_metrics.csv
    """
    hostname = socket.gethostname()
    host_dir = os.path.join(base_dir, hostname)
    os.makedirs(host_dir, exist_ok=True)

    # Find existing runs
    existing_runs = glob.glob(os.path.join(host_dir, "run_*_system_metrics.csv"))
    max_run = 0
    for path in existing_runs:
        try:
            # Extract N from .../run_N_system_metrics.csv
            filename = os.path.basename(path)
            parts = filename.split('_')
            # parts[0] is 'run', parts[1] is the number
            run_num = int(parts[1])
            if run_num > max_run:
                max_run = run_num
        except (IndexError, ValueError):
            continue

    next_run = max_run + 1
    
    sys_metric_path = os.path.join(host_dir, f"run_{next_run}_system_metrics.csv")
    train_metric_path = os.path.join(host_dir, f"run_{next_run}_training_metrics.csv")
    
    print(f"📁 Logging metrics to: {host_dir} (Run #{next_run})")
    return sys_metric_path, train_metric_path

def main():
    args = parse_arguments()

    RUN_ID = "tinyimagenet_resnet50"
    BATCH = args.batch_size
    TARGET_GLOBAL_BSZ = args.target_batch_size
    EPOCHS = args.epochs
    LR = args.lr
    WORKERS = 0          # En MPS con fork, 0 es OBLIGATORIO para evitar SegFaults.
    MATCHMAKING_TIME = 60.0
    AVERAGING_TIMEOUT = 120.0
    CHECKPOINT_DIR = "./checkpoints"

    # Initialize Loggers
    sys_log_path, train_log_path = get_next_log_paths()
    
    resource_monitor = ResourceMonitor(log_file=sys_log_path)
    resource_monitor.start()
    training_logger = TrainingLogger(log_file=train_log_path)

    # Device
    device = select_device(args.device)
    print(f"\nDevice: {device}")
    if device.type == "mps":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # DataLoaders (Tiny-ImageNet with ImageFolder)
    print(f"🚀 Loading Tiny-ImageNet from {args.data_dir}")
    
    num_classes = 200  # Tiny-ImageNet has 200 classes
    WORKERS = args.num_workers  # Use from args instead of hardcoded
    
    train_loader, val_loader = get_data_loaders(
        args.data_dir,
        batch_size=BATCH,
        num_workers=WORKERS
    )


    # Modelo ResNet50 (1000 clases)
    # MASTER en CPU (para Hivemind)
    model = build_model(num_classes=num_classes)
    model = model.to("cpu")

    # SHADOW en Device (para cómputo MPS/GPU)
    model_on_device = build_model(num_classes=num_classes)
    model_on_device = model_on_device.to(device)

    # channels_last SOLO en CUDA (no en MPS por bug en backward)
    if device.type == "cuda":
        model_on_device = model_on_device.to(memory_format=torch.channels_last)

    # Optimizador base
    base_optimizer = torch.optim.Adam(model.parameters(), lr=LR)    

    # DHT
    dht_kwargs = dict(
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{args.host_port}"],
        start=True,
        await_ready=False
    )
    if args.initial_peer:
        dht_kwargs["initial_peers"] = [args.initial_peer]

    print(f"=== Hivemind DHT ===")
    dht = hivemind.DHT(**dht_kwargs)
    
    # Wait for DHT to be ready (with longer timeout than default 15s)
    print("⏳ Waiting for DHT to be ready...")
    try:
        dht.wait_until_ready(timeout=60.0)
        print("✅ DHT is ready!")
    except TimeoutError:
        print("⚠️  DHT timed out waiting for readiness. Continuing anyway (might fail later)...")
    
    # Mostrar info
    maddrs = [str(m) for m in dht.get_visible_maddrs()]
    print("\n=== Hivemind DHT ===")
    for m in maddrs:
        print("VISIBLE_MADDR:", m)
    if not args.initial_peer:
        print("\n⚠️  Usa una dirección que NO sea 127.0.0.1 como --initial_peer")

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

    # Scheduler
    # Scheduler (usamos base_optimizer porque Hivemind Optimizer envuelve los param_groups de forma compleja)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(base_optimizer, milestones=args.scheduler_milestones, gamma=args.scheduler_gamma)

    # --- LOAD CHECKPOINT AFTER OPT CREATION ---
    # --- LOAD CHECKPOINT AFTER OPT CREATION ---
    # Intentar cargar latest primero
    latest_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
    best_path = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt")
    
    if os.path.exists(latest_path):
        print(f"Intentando reanudar desde LATEST: {latest_path}")
        # IMPORTANTE: Pasamos 'opt' (Hivemind Optimizer) en lugar de 'base_optimizer'
        start_epoch, acc = load_checkpoint(latest_path, model, opt, scheduler, device)
        
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location='cpu')
            best_accuracy = checkpoint.get("val_accuracy", acc)
        else:
            best_accuracy = acc
    elif os.path.exists(best_path):
        print(f"Intentando reanudar desde BEST: {best_path}")
        start_epoch, best_accuracy = load_checkpoint(best_path, model, opt, scheduler, device)
    else:
        print("No se encontraron checkpoints para cargar.")
    # ------------------------------------------

    target_epochs = EPOCHS
    last_seen_epoch = getattr(opt, "local_epoch", 0)
    checkpoint_path = None
    
    # Training accuracy tracking
    train_correct = 0
    train_total = 0

    print(f"\nEntrenando hasta {target_epochs} épocas globales (target_batch_size={TARGET_GLOBAL_BSZ}).")
    if best_accuracy > 0:
        print(f"Continuando desde mejor accuracy: {best_accuracy:.2f}%")

    try:
        with tqdm(total=None) as pbar:
            while True:
                for xb, yb in train_loader:
                    nb = (device.type == "cuda")  # non_blocking solo CUDA

                    xb = xb.to(device, non_blocking=nb)
                    yb = yb.to(device, non_blocking=nb)

                    # Safety check for labels
                    if (yb < 0).any() or (yb >= num_classes).any():
                        invalid_vals = yb[(yb < 0) | (yb >= num_classes)]
                        raise RuntimeError(f"Found invalid labels in batch: {invalid_vals.cpu().numpy()}. Expected range [0, {num_classes}).")

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
                    
                    # Calculate training accuracy
                    with torch.no_grad():
                        pred = logits.argmax(dim=1)
                        train_correct += (pred == yb).sum().item()
                        train_total += yb.size(0)
                    
                    loss.backward()

                    # 3. Sincronizar gradientes Device -> CPU
                    with torch.no_grad():
                        for p_cpu, p_dev in zip(model.parameters(), model_on_device.parameters()):
                            if p_dev.grad is not None:
                                if p_cpu.grad is None:
                                    p_cpu.grad = torch.zeros_like(p_cpu)
                                p_cpu.grad.copy_(p_dev.grad)

                    # 3.5 Sincronizar buffers Device -> CPU (CRITICAL for BatchNorm)
                    with torch.no_grad():
                        for b_cpu, b_dev in zip(model.buffers(), model_on_device.buffers()):
                            b_cpu.copy_(b_dev)

                    # 4. Step en CPU (Hivemind)
                    opt.step()
                    opt.zero_grad()
                    
                    # Calculate current training accuracy
                    current_train_acc = 100.0 * train_correct / max(1, train_total)

                    pbar.set_description(
                        f"loss={loss.item():.4f}  train_acc={current_train_acc:.2f}%  epoch_g={getattr(opt,'local_epoch',0)}  best_val={best_accuracy:.2f}%"
                    )
                    pbar.update()
                    
                    # Log training step
                    current_lr = scheduler.get_last_lr()[0]
                    training_logger.log_step(
                        epoch=getattr(opt, "local_epoch", 0),
                        batch=pbar.n, # approximate batch count
                        loss=loss.item(),
                        learning_rate=current_lr,
                        accuracy=current_train_acc  # Log training accuracy
                    )

                    current_epoch = getattr(opt, "local_epoch", last_seen_epoch)
                    if current_epoch != last_seen_epoch:
                        # Print final training accuracy for completed epoch
                        final_train_acc = 100.0 * train_correct / max(1, train_total)
                        tqdm.write(f"[Época {last_seen_epoch}] Training accuracy: {final_train_acc:.2f}%")
                        
                        # Reset training accuracy counters for new epoch
                        train_correct = 0
                        train_total = 0
                        
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

                            val_loss, val_acc = evaluate_accuracy(model_on_device, val_loader, device, max_batches=args.val_batches)
                            tqdm.write(f"[Época {current_epoch}] Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
                            
                            # Log validation metrics
                            training_logger.log_step(
                                epoch=current_epoch,
                                batch=pbar.n,
                                loss=val_loss,  # Log validation loss
                                learning_rate=scheduler.get_last_lr()[0],
                                accuracy=val_acc
                            )

                            # SIEMPRE guardar el latest checkpoint (usando opt, no base_optimizer)
                            save_checkpoint(model, opt, scheduler, CHECKPOINT_DIR, current_epoch, val_acc, filename="latest_checkpoint.pt")
                            
                            if val_acc > best_accuracy:
                                ckpt_path = save_checkpoint(model, opt, scheduler, CHECKPOINT_DIR, current_epoch, val_acc, filename="best_checkpoint.pt")
                                best_accuracy = val_acc
                                checkpoint_path = ckpt_path
                                tqdm.write(f"↑ Nuevo mejor accuracy ({best_accuracy:.2f}%). Checkpoint guardado: {ckpt_path}")
                            else:
                                tqdm.write(f"↔ No mejora (best={best_accuracy:.2f}%). Guardado latest_checkpoint.pt.")

                        last_seen_epoch = current_epoch
                        scheduler.step()

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

    resource_monitor.stop()

if __name__ == "__main__":
    main()
