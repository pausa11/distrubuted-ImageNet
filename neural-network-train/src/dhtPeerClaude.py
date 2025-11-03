import os
import argparse
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __name__ == "__main__":
    mp.set_start_method('fork', force=True)

import hivemind

def build_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 5), nn.MaxPool2d(2, 2), nn.ReLU(),
        nn.Conv2d(16, 32, 5), nn.MaxPool2d(2, 2), nn.ReLU(),
        nn.Flatten(), nn.Linear(32 * 5 * 5, 10)
    )

def get_dataloaders(data_root: str, batch: int, workers: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainDataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    validationDataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        trainDataset,
        shuffle=True,
        batch_size=batch,
        num_workers=workers,
        pin_memory=False,
        drop_last=False,
    )
    
    validation_loader = DataLoader(
        validationDataset,
        shuffle=False,
        batch_size=max(256, batch),
        num_workers=min(2, workers),
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, validation_loader

def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    accuracy = 100.0 * correct / max(1, total)
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
    p = argparse.ArgumentParser(description="CIFAR-10 x Hivemind - Versión Optimizada")
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None, 
                   help="Forzar dispositivo (cpu o cuda). Por defecto: auto-detectar")
    p.add_argument("--use_checkpoint", action="store_true", 
                   help="Cargar checkpoint si existe en ./checkpoints/best_checkpoint.pt")
    p.add_argument("--initial_peer", type=str, default=None, 
                   help="Multiaddr de un peer inicial para bootstrap (opcional en el primer nodo)")
    
    # NUEVOS PARÁMETROS DE OPTIMIZACIÓN
    p.add_argument("--target_batch_size", type=int, default=50000, 
                   help="Batch global objetivo (default: 50000 = 1 época completa)")
    p.add_argument("--local_batch_size", type=int, default=64, 
                   help="Batch size local (default: 64, más grande = menos overhead)")
    p.add_argument("--matchmaking_time", type=float, default=1.5, 
                   help="Tiempo de búsqueda de peers en segundos (default: 1.5)")
    p.add_argument("--averaging_timeout", type=float, default=5.0, 
                   help="Timeout de sincronización en segundos (default: 5.0)")
    p.add_argument("--num_epochs", type=int, default=50, 
                   help="Número de épocas globales (default: 50)")
    p.add_argument("--compression", action="store_true", 
                   help="Usar compresión FP16 para reducir transferencia de red")
    
    return p.parse_args()

def main():
    args = parse_arguments()

    # Configuración
    DATA_ROOT = "./data"
    RUN_ID = "cifar_optimized"
    LR = 1e-3
    MOMENTUM = 0.9
    WORKERS = 2
    CHECKPOINT_DIR = "./checkpoints"

    # Usar parámetros optimizados
    BATCH = args.local_batch_size
    TARGET_GLOBAL_BSZ = args.target_batch_size
    EPOCHS = args.num_epochs
    MATCHMAKING_TIME = args.matchmaking_time
    AVERAGING_TIMEOUT = args.averaging_timeout

    print("\n" + "="*60)
    print("CONFIGURACIÓN DE OPTIMIZACIÓN")
    print("="*60)
    print(f"Batch local:              {BATCH}")
    print(f"Target batch global:      {TARGET_GLOBAL_BSZ}")
    print(f"Épocas objetivo:          {EPOCHS}")
    print(f"Matchmaking time:         {MATCHMAKING_TIME}s")
    print(f"Averaging timeout:        {AVERAGING_TIMEOUT}s")
    print(f"Compresión FP16:          {'✓ Activada' if args.compression else '✗ Desactivada'}")
    
    # Calcular sincronizaciones esperadas
    syncs_per_epoch = 50000 // TARGET_GLOBAL_BSZ
    total_syncs = EPOCHS * syncs_per_epoch
    estimated_comm_time = total_syncs * (MATCHMAKING_TIME + 1.5)  # +1.5s promedio de all-reduce
    print(f"\nSincronizaciones/época:   {syncs_per_epoch}")
    print(f"Total sincronizaciones:   {total_syncs}")
    print(f"Tiempo comunicación est.: {estimated_comm_time:.1f}s ({estimated_comm_time/60:.1f} min)")
    print("="*60 + "\n")

    train_loader, validation_loader = get_dataloaders(DATA_ROOT, BATCH, WORKERS)

    model = build_model()
    
    # Configurar DHT
    dht_kwargs = dict(
        host_maddrs=["/ip4/0.0.0.0/tcp/0"],
        start=True
    )
    if args.initial_peer:
        dht_kwargs["initial_peers"] = [args.initial_peer]

    dht = hivemind.DHT(**dht_kwargs)
    
    maddrs = [str(m) for m in dht.get_visible_maddrs()]
    print("=== Hivemind DHT ===")
    for m in maddrs:
        print("VISIBLE_MADDR:", m)
    if not args.initial_peer:
        print("\n⚠️  Usa una dirección que NO sea 127.0.0.1 como --initial_peer")
        print("    Busca la que tenga tu IP local (192.168.x.x o 10.0.x.x)")

    # Configurar device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    print(f"\nDevice: {device}")
    
    # Crear el optimizador base
    base_optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    # Cargar checkpoint si se solicita
    best_accuracy = -1.0
    start_epoch = 0
    if args.use_checkpoint:
        ckpt_path = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pt")
        start_epoch, best_accuracy = load_checkpoint(ckpt_path, model, base_optimizer, device)

    # OPTIMIZACIÓN: Configurar compresión si está habilitada
    compression_config = None
    if args.compression:
        try:
            compression_config = hivemind.Compression(
                compression=hivemind.CompressionType.FLOAT16,
                min_compression_ratio=1.5
            )
            print("✓ Compresión FP16 habilitada")
        except:
            print("⚠️  Compresión no disponible en esta versión de Hivemind")

    # Crear el optimizador de Hivemind con configuración optimizada
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

    # Configurar pin_memory si hay GPU
    if device.type == "cuda":
        train_loader.pin_memory = True
        validation_loader.pin_memory = True

    target_epochs = EPOCHS
    last_seen_epoch = getattr(opt, "local_epoch", 0)
    checkpoint_path = None

    print(f"\nEntrenando hasta {target_epochs} épocas globales (target_batch_size={TARGET_GLOBAL_BSZ}).")
    if args.use_checkpoint and best_accuracy > 0:
        print(f"Continuando desde mejor accuracy: {best_accuracy:.2f}%")

    # OPTIMIZACIÓN: Contador de sincronizaciones
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
                    
                    # OPTIMIZACIÓN: Medir tiempo de step
                    step_start = time.time()
                    opt.step()
                    step_time = time.time() - step_start

                    pbar.set_description(
                        f"loss={loss.item():.4f}  epoch_g={getattr(opt,'local_epoch',0)}  "
                        f"best={best_accuracy:.2f}%  step={step_time:.2f}s"
                    )
                    pbar.update()

                    # Detectar transición de época global
                    current_epoch = getattr(opt, "local_epoch", last_seen_epoch)
                    if current_epoch != last_seen_epoch:
                        sync_count += 1
                        elapsed = time.time() - start_time
                        
                        # Evaluar en validación
                        val_acc = evaluate_accuracy(model, validation_loader, device)
                        tqdm.write(
                            f"[Época {current_epoch}] Accuracy: {val_acc:.2f}%  "
                            f"Syncs: {sync_count}  Tiempo: {elapsed:.1f}s"
                        )

                        # Guardar solo si mejora
                        if val_acc > best_accuracy:
                            ckpt_path = save_checkpoint(model, base_optimizer, CHECKPOINT_DIR, current_epoch, val_acc)
                            best_accuracy = val_acc
                            checkpoint_path = ckpt_path
                            tqdm.write(f"↑ Nuevo mejor accuracy ({best_accuracy:.2f}%). Checkpoint guardado.")
                        
                        last_seen_epoch = current_epoch

                        # Condición de parada
                        if current_epoch >= target_epochs:
                            tqdm.write(f"✓ Alcanzadas {current_epoch} épocas globales. Terminando...")
                            raise StopIteration
    except StopIteration:
        pass

    # Reporte final con estadísticas
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("RESULTADOS FINALES")
    print("="*60)
    print(f"Mejor accuracy:           {best_accuracy:.2f}%")
    print(f"Tiempo total:             {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total sincronizaciones:   {sync_count}")
    print(f"Tiempo por sync:          {total_time/max(1, sync_count):.1f}s")
    if checkpoint_path:
        print(f"Mejor checkpoint:         {checkpoint_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()