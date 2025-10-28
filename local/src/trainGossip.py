import argparse, os, re
import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method('fork', force=True)

import torch, torch.nn as nn
from torch.utils.data import DataLoader
import hivemind

from datasets import get_tiny_imagenet
from model import resnet50_tiny
from metrics import eval_top1_acc
from controller_client import Reporter

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/tiny-imagenet-200")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--target_global_bsz", type=int, default=512)
    ap.add_argument("--initial_peer", type=str, default=None, help="multiaddr completa con /p2p/<peerID>")
    ap.add_argument("--host_maddr", type=str, default=None, help="p.ej. /ip4/127.0.0.1/tcp/4011 (opcional)")
    ap.add_argument("--run_id", type=str, default="tiny-imagenet-gossip")
    ap.add_argument("--controller", type=str, default=None)
    ap.add_argument("--val_interval", type=int, default=500)
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    train_ds, val_ds = get_tiny_imagenet(args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=256,       num_workers=4, pin_memory=True)

    # Modelo en CPU (como sugiere el tutorial)
    model = resnet50_tiny(num_classes=200)
    criterion = nn.CrossEntropyLoss()
    base_opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # DHT: deja que el daemon elija puerto si no pasas host_maddr
    dht_kwargs = {"start": True}
    if args.host_maddr:
        dht_kwargs["host_maddrs"] = [args.host_maddr]
    if args.initial_peer:
        dht_kwargs["initial_peers"] = [args.initial_peer]

    dht = hivemind.DHT(**dht_kwargs)

    # Imprime visible maddrs (con /p2p/<peerID>) para que otros se conecten
    visible = [str(m) for m in dht.get_visible_maddrs()]
    print("VISIBLE_MADDRS:", visible)
    if not args.initial_peer:
        print("Comparte una de estas como --initial_peer en otros peers ↑")

    # Optimizer descentralizado (crear ANTES de mover a GPU)
    opt = hivemind.Optimizer(
        dht=dht,
        run_id=args.run_id,
        optimizer=base_opt,
        batch_size_per_step=args.batch,
        target_batch_size=args.target_global_bsz,
        use_local_updates=True,
        matchmaking_time=3.0,
        averaging_timeout=10.0,
        verbose=True
    )

    # Ahora sí, mueve a GPU si hay
    if device == "cuda":
        model.cuda()

    # Descarga estado más reciente para no "gastar" el primer step
    try:
        opt.load_state_from_peers()
    except Exception as e:
        print("Aviso: no se pudo cargar estado inicial de peers (aún).", e)

    peer_id = visible[0] if visible else (args.host_maddr or "peer-unknown")
    reporter = Reporter(args.controller, peer_id) if args.controller else None

    # Entrenamiento
    global_step, samples_seen = 0, 0
    model.train()
    for epoch in range(args.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            global_step += 1
            samples_seen += x.size(0)

            if reporter and (global_step % 20 == 0):
                lr = next(g['lr'] for g in opt.local_optimizer.param_groups)
                reporter.send(global_step, float(loss.item()), lr, int(samples_seen))

            if (global_step % args.val_interval) == 0:
                acc = eval_top1_acc(model, val_loader, device=device)
                if reporter:
                    lr = next(g['lr'] for g in opt.local_optimizer.param_groups)
                    reporter.send(global_step, float(loss.item()), lr, int(samples_seen), acc_val_top1=float(acc))
                print(f"[{peer_id}] step={global_step} val@top1={acc:.3f}")

    # Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    port_str = "ephemeral"
    if args.host_maddr:
        m = re.search(r"/tcp/(\d+)", args.host_maddr)
        if m: port_str = m.group(1)
    torch.save(model.state_dict(), f"checkpoints/resnet50_tiny_{port_str}.pt")

if __name__ == "__main__":
    main()