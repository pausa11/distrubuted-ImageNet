from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import time

app = FastAPI()
STATE: Dict[str, dict] = {}

class Report(BaseModel):
    peer_id: str
    step: int
    loss: float
    lr: float
    samples_seen: int
    avg_interval_s: float
    acc_val_top1: float | None = None
    ts: float = time.time()

@app.post("/report")
def report_metrics(r: Report):
    STATE[r.peer_id] = r.dict()
    return {"ok": True}

@app.get("/state")
def get_state():
    return STATE
