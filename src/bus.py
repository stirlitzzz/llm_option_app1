# src/bus.py
from __future__ import annotations
from typing import Any, Optional, TypedDict
import json, os, time, uuid, threading

class BusRecord(TypedDict, total=False):
    run_id: str
    topic: str
    key: str
    payload: Any
    producer: str
    ts: float

class Bus:
    """Tiny in-process KV + JSONL log for agent handoffs."""
    def __init__(self, path: str = "data", log_name: Optional[str] = None):
        os.makedirs(path, exist_ok=True)
        self.kv_path = os.path.join(path, "kv.json")
        self.log_path = os.path.join(path, log_name or time.strftime("bus-%Y%m%d.jsonl"))
        self._lock = threading.Lock()
        self._kv: dict[str, Any] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.kv_path):
            try:
                with open(self.kv_path, "r") as f:
                    self._kv = json.load(f)
            except Exception:
                self._kv = {}

    def _persist(self):
        with open(self.kv_path, "w") as f:
            json.dump(self._kv, f, separators=(",", ":"))

    def put(self, *, topic: str, key: str, payload: Any, producer: str, run_id: Optional[str] = None) -> BusRecord:
        rec: BusRecord = {
            "run_id": run_id or f"{int(time.time())}-{uuid.uuid4().hex[:6]}",
            "topic": topic,
            "key": key,
            "payload": payload,
            "producer": producer,
            "ts": time.time(),
        }
        with self._lock:
            self._kv[f"{topic}:{key}"] = payload
            self._persist()
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def get(self, *, topic: str, key: str, default=None):
        return self._kv.get(f"{topic}:{key}", default)

    def list_keys(self, topic: str) -> list[str]:
        prefix = f"{topic}:"
        return [k.split(":", 1)[1] for k in self._kv.keys() if k.startswith(prefix)]

# singleton shared across processes (import by tools)
BUS = Bus(path="data")
