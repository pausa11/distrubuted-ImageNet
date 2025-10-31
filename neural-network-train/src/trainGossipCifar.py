import argparse, os, re
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('fork', force=True)

import torch, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import hivemind

from controller_client import Reporter

def get_cifar10(data_root="./data"):
    """Descarga y prepara CIFAR-10"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_ds = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    val_ds = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    
    return train_ds, val_ds

def get_simple_cnn():
    """CNN simple para CIFAR-10 (10 clases)"""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 10)
    )

def eval_top1_acc(model, loader, device):
    """Evalúa accuracy top-1"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total if total > 0 else 0.0

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--target_global_bsz", type=int, default=512)
    ap.add_argument("--initial_peer", type=str, default=None, help="multiaddr completa con /p2p/<peerID>")
    ap.add_argument("--host_maddr", type=str, default=None, help="p.ej. /ip4/127.0.0.1/tcp/4011 (opcional)")
    ap.add_argument("--run_id", type=str, default="cifar10-gossip")
    ap.add_argument("--controller", type=str, default=None)
    ap.add_argument("--val_interval", type=int, default=100)
    ap.add_argument("--resume_from", type=str, default=None, help="Path al checkpoint para reanudar entrenamiento")
    ap.add_argument("--warmup_epochs", type=int, default=1, help="Número de epochs para warmup del LR")
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset CIFAR-10
    print("Descargando CIFAR-10...")
    train_ds, val_ds = get_cifar10(args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, num_workers=2, pin_memory=True)
    print(f"Train: {len(train_ds)} imágenes, Val: {len(val_ds)} imágenes")

    # Modelo simple
    model = get_simple_cnn()
    criterion = nn.CrossEntropyLoss()
    base_opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # Learning rate schedulers con warmup y cosine annealing
    # Calculamos steps basados en train_loader (considera drop_last=True)
    steps_per_epoch = len(train_loader)
    total_training_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    
    # Usamos SequentialLR para combinar warmup + cosine annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        base_opt, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_opt, 
        T_max=total_training_steps - warmup_steps, 
        eta_min=0.001
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        base_opt,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # Cargar checkpoint si se especifica
    start_step = 0
    best_acc = 0.0
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"📂 Cargando checkpoint desde: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            base_opt.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ Scheduler state cargado")
            start_step = checkpoint.get('step', 0)
            best_acc = checkpoint.get('best_acc', 0.0)
            print(f"✅ Checkpoint cargado! Step: {start_step}, Best acc: {best_acc:.3f}")
        else:
            print(f"⚠️  Checkpoint no encontrado: {args.resume_from}")
            print("   Iniciando desde cero...")
    else:
        print("🆕 Iniciando entrenamiento desde cero")

    # DHT: deja que el daemon elija puerto si no pasas host_maddr
    dht_kwargs = {"start": True}
    if args.host_maddr:
        dht_kwargs["host_maddrs"] = [args.host_maddr]
    if args.initial_peer:
        dht_kwargs["initial_peers"] = [args.initial_peer]

    dht = hivemind.DHT(**dht_kwargs)

    # Imprime visible maddrs (con /p2p/<peerID>) para que otros se conecten
    visible = [str(m) for m in dht.get_visible_maddrs()]
    print("=" * 60)
    print("VISIBLE_MADDRS:", visible)
    if not args.initial_peer:
        print("✨ Comparte una de estas como --initial_peer en otros peers ↑")
    print("=" * 60)

    # Optimizer descentralizado (crear ANTES de mover a GPU)
    opt = hivemind.Optimizer(
        dht=dht,
        run_id=args.run_id,
        optimizer=base_opt,
        batch_size_per_step=args.batch,
        target_batch_size=args.target_global_bsz,
        use_local_updates=True,
        matchmaking_time=5.0,
        averaging_timeout=15.0,
        verbose=True,
        grad_compression=hivemind.compression.Float16Compression(),
    )

    # Ahora sí, mueve a GPU si hay
    if device == "cuda":
        model.cuda()
        print(f"🚀 Usando GPU: {torch.cuda.get_device_name(0)}")

    # Descarga estado más reciente para no "gastar" el primer step
    try:
        opt.load_state_from_peers()
        print("✅ Estado inicial cargado desde peers")
    except Exception as e:
        print(f"⚠️  No se pudo cargar estado inicial de peers: {e}")

    peer_id = visible[0] if visible else (args.host_maddr or "peer-unknown")
    reporter = Reporter(args.controller, peer_id) if args.controller else None

    # Entrenamiento
    print(f"\n🎯 Iniciando entrenamiento ({args.epochs} epochs, batch={args.batch}, target_bsz={args.target_global_bsz})")
    print(f"   LR inicial: {args.lr}, LR final: 0.001, Warmup epochs: {args.warmup_epochs}")
    if start_step > 0:
        print(f"   Reanudando desde step {start_step} con best_acc={best_acc:.3f}")
    global_step, samples_seen = start_step, start_step * args.batch
    model.train()
    
    for epoch in range(args.epochs):
        print(f"\n📊 Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            
            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            opt.step()
            
            # Actualizar scheduler después de cada step
            scheduler.step()

            global_step += 1
            samples_seen += x.size(0)

            if batch_idx % 20 == 0:
                current_lr = base_opt.param_groups[0]['lr']
                print(f"  Step {global_step} | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | Samples: {samples_seen}")

            if reporter and (global_step % 20 == 0):
                lr = next(g['lr'] for g in base_opt.param_groups)
                reporter.send(global_step, float(loss.item()), lr, int(samples_seen))

            if (global_step % args.val_interval) == 0:
                acc = eval_top1_acc(model, val_loader, device=device)
                if reporter:
                    lr = next(g['lr'] for g in base_opt.param_groups)
                    reporter.send(global_step, float(loss.item()), lr, int(samples_seen), acc_val_top1=float(acc))
                print(f"🎯 [{peer_id}] step={global_step} val_acc={acc:.3f}")
                
                # Guardar si es el mejor modelo
                if acc > best_acc:
                    best_acc = acc
                    os.makedirs("checkpoints", exist_ok=True)
                    port_str = "ephemeral"
                    if args.host_maddr:
                        m = re.search(r"/tcp/(\d+)", args.host_maddr)
                        if m: port_str = m.group(1)
                    checkpoint_path = f"checkpoints/cifar10_cnn_{port_str}_best.pt"
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': base_opt.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_acc,
                    }, checkpoint_path)
                    print(f"💾 Nuevo mejor modelo guardado! Acc: {best_acc:.3f} → {checkpoint_path}")

    # Evaluación final
    final_acc = eval_top1_acc(model, val_loader, device=device)
    print(f"\n✅ Entrenamiento completado!")
    print(f"   Accuracy final: {final_acc:.3f}")
    print(f"   Mejor accuracy: {best_acc:.3f}")

    # Checkpoint final (último estado)
    os.makedirs("checkpoints", exist_ok=True)
    port_str = "ephemeral"
    if args.host_maddr:
        m = re.search(r"/tcp/(\d+)", args.host_maddr)
        if m: port_str = m.group(1)
    checkpoint_path = f"checkpoints/cifar10_cnn_{port_str}_final.pt"
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': base_opt.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_acc': final_acc,
        'best_acc': best_acc,
    }, checkpoint_path)
    print(f"💾 Modelo final guardado en: {checkpoint_path}")

if __name__ == "__main__":
    main()