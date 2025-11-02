import os
import argparse
import multiprocessing as mp
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __name__ == "__main__":
    # En Unix/macOS evita problemas con procesos auxiliares
    mp.set_start_method('fork', force=True)

import hivemind


def build_model():
    # CNN simple para CIFAR-10
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

    trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    valset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        trainset,
        shuffle=True,
        batch_size=batch,
        num_workers=workers,
        pin_memory=False,  # se activará luego si usamos CUDA
        drop_last=False,
    )
    # Para validación no necesitamos barajar
    val_loader = DataLoader(
        valset,
        shuffle=False,
        batch_size=max(256, batch),  # batch grande para validar más rápido
        num_workers=min(2, workers),
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader


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
    acc = 100.0 * correct / max(1, total)
    model.train()
    return acc


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, out_dir: str, epoch_idx: int, acc: float):
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"ckpt_epoch{epoch_idx:04d}_acc{acc:.2f}_{stamp}.pt")
    torch.save({
        "epoch": epoch_idx,
        "val_accuracy": acc,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
    }, path)
    return path


def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 x Hivemind (guardar checkpoint solo si mejora accuracy)")
    p.add_argument("--data_root", type=str, default="./data", help="Ruta al directorio de datos")
    p.add_argument("--run_id", type=str, default="my_cifar_run", help="ID lógico de la corrida colaborativa")
    p.add_argument("--initial_peer", type=str, default=None, help="Multiaddr de un peer inicial para bootstrap (opcional en el primer nodo)")
    p.add_argument("--host", type=str, default=None, help="Host local para el DHT (opcional)")
    p.add_argument("--port", type=int, default=None, help="Puerto local para el DHT (opcional)")

    p.add_argument("--batch", type=int, default=32, help="Batch local por step (coincide con batch_size_per_step)")
    p.add_argument("--target_global_bsz", type=int, default=10_000, help="Muestras globales para cerrar una época")
    p.add_argument("--epochs", type=int, default=5, help="Número de épocas globales a entrenar antes de parar")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)

    p.add_argument("--workers", type=int, default=2, help="num_workers del DataLoader")
    p.add_argument("--matchmaking_time", type=float, default=3.0)
    p.add_argument("--averaging_timeout", type=float, default=10.0)

    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    return p.parse_args()


def main():
    args = parse_args()

    # ===== 1) Dataloaders =====
    train_loader, val_loader = get_dataloaders(args.data_root, args.batch, args.workers)

    # ===== 2) Modelo + SGD (CPU por ahora) =====
    model = build_model()
    base_opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # ===== 3) DHT (P2P discovery) =====
    dht_kwargs = dict(start=True)
    if args.host: dht_kwargs["host"] = args.host
    if args.port: dht_kwargs["port"] = args.port
    if args.initial_peer:
        dht_kwargs["initial_peers"] = [args.initial_peer]

    dht = hivemind.DHT(**dht_kwargs)
    maddrs = [str(m) for m in dht.get_visible_maddrs()]
    print("\n=== Hivemind DHT ===")
    for m in maddrs:
        print("VISIBLE_MADDR:", m)
    if not args.initial_peer:
        print("Comparte uno de estos como --initial_peer en otros peers ↑")

    # ===== 4) Hivemind.Optimizer (aún en CPU) =====
    opt = hivemind.Optimizer(
        dht=dht,
        run_id=args.run_id,
        batch_size_per_step=args.batch,
        target_batch_size=args.target_global_bsz,
        optimizer=base_opt,
        use_local_updates=True,
        matchmaking_time=args.matchmaking_time,
        averaging_timeout=args.averaging_timeout,
        verbose=True,
    )

    # ===== 5) Device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        train_loader.pin_memory = True
        val_loader.pin_memory = True

    # ===== 6) Entrenamiento con validación por época global =====
    target_epochs = args.epochs
    last_seen_epoch = getattr(opt, "local_epoch", 0)
    best_acc = -1.0
    best_ckpt_path = None

    print(f"\nEntrenando hasta {target_epochs} épocas globales (target_batch_size={args.target_global_bsz}).")
    print("Device:", device)

    try:
        with tqdm(total=None) as pbar:
            while True:
                for xb, yb in train_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    opt.zero_grad()
                    loss = F.cross_entropy(model(xb), yb)
                    loss.backward()
                    opt.step()  # aplica SGD local y coordina averaging en background

                    pbar.set_description(f"loss={loss.item():.4f}  epoch_g={getattr(opt,'local_epoch',0)}  best={best_acc:.2f}%")
                    pbar.update()

                    # Detectar transición de época global
                    current_epoch = getattr(opt, "local_epoch", last_seen_epoch)
                    if current_epoch != last_seen_epoch:
                        # === Fin de época global: evaluar en validación ===
                        val_acc = evaluate_accuracy(model, val_loader, device)
                        tqdm.write(f"[Época {current_epoch}] Accuracy validación: {val_acc:.2f}%")

                        # Guardar solo si mejora
                        if val_acc > best_acc:
                            ckpt_path = save_checkpoint(model, base_opt, args.checkpoint_dir, current_epoch, val_acc)
                            best_acc = val_acc
                            best_ckpt_path = ckpt_path
                            tqdm.write(f"↑ Nuevo mejor accuracy ({best_acc:.2f}%). Checkpoint guardado: {ckpt_path}")
                        else:
                            tqdm.write(f"↔ No mejora (best={best_acc:.2f}%). No se guarda checkpoint.")

                        last_seen_epoch = current_epoch

                        # Condición de parada
                        if current_epoch >= target_epochs:
                            tqdm.write(f"✓ Alcanzadas {current_epoch} épocas globales. Terminando...")
                            raise StopIteration  # salir de ambos bucles
    except StopIteration:
        pass

    # Reporte final
    if best_ckpt_path:
        print(f"Entrenamiento finalizado. Mejor accuracy: {best_acc:.2f}%")
        print(f"Mejor checkpoint: {best_ckpt_path}")
    else:
        print("Entrenamiento finalizado. No se guardaron checkpoints (no hubo mejora).")


if __name__ == "__main__":
    main()
