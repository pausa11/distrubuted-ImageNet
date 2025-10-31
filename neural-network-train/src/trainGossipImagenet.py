import argparse, os, re
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('fork', force=True)

import torch, torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import hivemind

from controller_client import Reporter

def get_imagenet_tiny(data_root="./data"):
    """Carga ImageNet Tiny 200 desde carpeta local"""
    # Estadísticas de ImageNet (las más comunes para ImageNet Tiny)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        normalize,
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Asume estructura: data/tiny-imagenet-200/train/ y data/tiny-imagenet-200/val/
    train_path = os.path.join(data_root, "tiny-imagenet-200", "train")
    val_path = os.path.join(data_root, "tiny-imagenet-200", "val")
    
    train_ds = ImageFolder(root=train_path, transform=transform_train)
    val_ds = ImageFolder(root=val_path, transform=transform_test)
    
    return train_ds, val_ds

def get_resnet50(num_classes=200, pretrained=False):
    """ResNet50 adaptado para ImageNet Tiny (200 clases)"""
    model = torchvision.models.resnet50(pretrained=pretrained)
    
    # Modificar la capa final para 200 clases
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

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
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--target_global_bsz", type=int, default=256)
    ap.add_argument("--initial_peer", type=str, default=None, help="multiaddr completa con /p2p/<peerID>")
    ap.add_argument("--host_maddr", type=str, default=None, help="p.ej. /ip4/127.0.0.1/tcp/4011 (opcional)")
    ap.add_argument("--run_id", type=str, default="imagenet-tiny-gossip")
    ap.add_argument("--controller", type=str, default=None)
    ap.add_argument("--val_interval", type=int, default=500)
    ap.add_argument("--resume_from", type=str, default=None, help="Path al checkpoint para reanudar entrenamiento")
    ap.add_argument("--pretrained", action="store_true", help="Usar pesos pre-entrenados de ImageNet")
    
    # Gossip algorithm parameters
    ap.add_argument("--gossip_group_size", type=int, default=16, help="Tamaño del grupo de gossip (potencia de 2 recomendada)")
    ap.add_argument("--gossip_min_group_size", type=int, default=2, help="Tamaño mínimo del grupo para comenzar averaging")
    ap.add_argument("--gossip_alpha", type=float, default=1.0, help="Tasa de aprendizaje para averaging (1.0 = promedio completo)")
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset ImageNet Tiny 200
    print("Cargando ImageNet Tiny 200...")
    train_ds, val_ds = get_imagenet_tiny(args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, num_workers=4, pin_memory=True)
    print(f"Train: {len(train_ds)} imágenes, Val: {len(val_ds)} imágenes")

    # Modelo ResNet50
    model = get_resnet50(num_classes=200, pretrained=args.pretrained)
    criterion = nn.CrossEntropyLoss()
    base_opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # Cargar checkpoint si se especifica
    start_step = 0
    best_acc = 0.0
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"📂 Cargando checkpoint desde: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            base_opt.load_state_dict(checkpoint['optimizer_state_dict'])
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

    # Configuración del algoritmo Gossip para averaging descentralizado
    # El algoritmo Gossip permite que los peers se conecten y promedien sus parámetros
    # en grupos pequeños de forma descentralizada, sin necesidad de un coordinador central.
    # Esto mejora la escalabilidad y la tolerancia a fallos.
    
    gossip_averager_opts = {
        "target_group_size": args.gossip_group_size,  # Tamaño del grupo de gossip
        "min_group_size": args.gossip_min_group_size,  # Mínimo de peers para comenzar
        "averaging_alpha": args.gossip_alpha,  # Factor de averaging (1.0 = promedio completo)
        "min_matchmaking_time": 5.0,  # Tiempo mínimo para encontrar peers
    }
    
    # Optimizer descentralizado con algoritmo Gossip (crear ANTES de mover a GPU)
    # use_local_updates=True habilita el modo de "local SGD" donde cada peer entrena
    # localmente y periódicamente promedia sus parámetros con otros peers usando Gossip
    opt = hivemind.Optimizer(
        dht=dht,
        run_id=args.run_id,
        optimizer=base_opt,
        batch_size_per_step=args.batch,
        target_batch_size=args.target_global_bsz,
        use_local_updates=True,  # Habilita modo Local SGD con Gossip averaging
        matchmaking_time=3.0,
        averaging_timeout=10.0,
        averager_opts=gossip_averager_opts,  # Configuración explícita del Gossip
        verbose=True
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
            opt.step()

            global_step += 1
            samples_seen += x.size(0)

            if batch_idx % 20 == 0:
                print(f"  Step {global_step} | Loss: {loss.item():.4f} | Samples: {samples_seen}")

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
                    checkpoint_path = f"checkpoints/imagenet_tiny_resnet50_{port_str}_best.pt"
                    torch.save({
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': base_opt.state_dict(),
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
    checkpoint_path = f"checkpoints/imagenet_tiny_resnet50_{port_str}_final.pt"
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': base_opt.state_dict(),
        'final_acc': final_acc,
        'best_acc': best_acc,
    }, checkpoint_path)
    print(f"💾 Modelo final guardado en: {checkpoint_path}")

if __name__ == "__main__":
    main()