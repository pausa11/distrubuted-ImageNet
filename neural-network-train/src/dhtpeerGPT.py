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


def get_dataloaders(data_root: str, batch: int, workers: int, device: torch.device):
    # Normalización simple ([-1,1]); si quieres las "oficiales" de CIFAR-10, avisa y te la cambio
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    val_dataset   = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    pin = (device.type == "cuda")
    # persistent_workers solo si workers > 0
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
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
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
    p = argparse.ArgumentParser(description="CIFAR-10 x Hivemind (optimizado para multi-peer)")
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None,
                   help="Forzar dispositivo (cpu o cuda). Por defecto: auto-detectar")
    p.add_argument("--use_checkpoint", action="store_true",
                   help="Cargar checkpoint si existe en ./checkpoints/best_checkpoint.pt")
    p.add_argument("--initial_peer", type=str, default=None,
                   help="Multiaddr de un peer inicial para bootstrap (opcional en el primer nodo)")
    p.add_argument("--val_every", type=int, default=5,
                   help="Frecuencia de validación (épocas globales). Ej: 5 = validar cada 5.")
    return p.parse_args()


def main():
    args = parse_arguments()

    # Configuración fija
    DATA_ROOT = "./data"
    RUN_ID = "my_cifar_run"
    BATCH = 32
    TARGET_GLOBAL_BSZ = 30_000  # ↑ menos rondas de averaging
    EPOCHS = 50
    LR = 1e-3
    MOMENTUM = 0.9
    WORKERS = 2
    MATCHMAKING_TIME = 1.5       # ↓ ventana de emparejamiento
    AVERAGING_TIMEOUT = 6.0      # ↓ timeout de averaging
    CHECKPOINT_DIR = "./checkpoints"

    # Configurar device primero (para crear DataLoaders con pin_memory correcto)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # DataLoaders (optimizado)
    train_loader, val_loader = get_dataloaders(DATA_ROOT, BATCH, WORKERS, device)

    # Modelo y optimizador base
    model = build_model().to(device)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    # DHT (escucha en todas las interfaces)
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

    # Cargar checkpoint si se solicita
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
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    opt.zero_grad()
                    loss = F.cross_entropy(model(xb), yb)
                    loss.backward()
                    opt.step()

                    pbar.set_description(
                        f"loss={loss.item():.4f}  epoch_g={getattr(opt,'local_epoch',0)}  best={best_accuracy:.2f}%"
                    )
                    pbar.update()

                    # Detectar transición de época global
                    current_epoch = getattr(opt, "local_epoch", last_seen_epoch)
                    if current_epoch != last_seen_epoch:
                        # Validar solo cada --val_every épocas (y siempre en la 1ra)
                        do_eval = (current_epoch == 1) or (current_epoch % args.val_every == 0) or (current_epoch >= target_epochs)
                        if do_eval:
                            val_acc = evaluate_accuracy(model, val_loader, device)
                            tqdm.write(f"[Época {current_epoch}] Accuracy validación: {val_acc:.2f}%")

                            # Guardar solo si mejora
                            if val_acc > best_accuracy:
                                ckpt_path = save_checkpoint(model, base_optimizer, CHECKPOINT_DIR, current_epoch, val_acc)
                                best_accuracy = val_acc
                                checkpoint_path = ckpt_path
                                tqdm.write(f"↑ Nuevo mejor accuracy ({best_accuracy:.2f}%). Checkpoint guardado: {ckpt_path}")
                            else:
                                tqdm.write(f"↔ No mejora (best={best_accuracy:.2f}%). No se guarda checkpoint.")

                        last_seen_epoch = current_epoch

                        # Condición de parada
                        if current_epoch >= target_epochs:
                            tqdm.write(f"✓ Alcanzadas {current_epoch} épocas globales. Terminando...")
                            raise StopIteration
    except StopIteration:
        pass

    # Reporte final
    if checkpoint_path or best_accuracy > 0:
        print(f"\nEntrenamiento finalizado. Mejor accuracy: {best_accuracy:.2f}%")
        if checkpoint_path:
            print(f"Mejor checkpoint: {checkpoint_path}")
    else:
        print("\nEntrenamiento finalizado. No se guardaron checkpoints (no hubo mejora).")


if __name__ == "__main__":
    main()
