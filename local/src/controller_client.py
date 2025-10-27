import time, requests

def safe_post(url, json, timeout=2.0):
    try:
        requests.post(url, json=json, timeout=timeout)
    except Exception:
        pass  # no interrumpas el entrenamiento por el monitor

class Reporter:
    def __init__(self, controller_url: str, peer_id: str):
        self.url = controller_url.rstrip("/")
        self.peer_id = peer_id
        self.last = time.time()

    def send(self, step, loss, lr, samples_seen, acc_val_top1=None):
        now = time.time()
        safe_post(f"{self.url}/report", {
            "peer_id": self.peer_id,
            "step": step,
            "loss": float(loss),
            "lr": float(lr),
            "samples_seen": int(samples_seen),
            "avg_interval_s": float(now - self.last),
            "acc_val_top1": None if acc_val_top1 is None else float(acc_val_top1),
            "ts": now
        })
        self.last = now
