# src/tools_bus.py
from agents import function_tool
from tracing import trace_tool
from bus import BUS
import json

@function_tool  # <- no strict kwarg
@trace_tool("bus_put")
def bus_put(topic: str, key: str, payload_json: str, producer: str, run_id: str | None = None) -> dict:
    """
    payload_json: JSON string (dict-like). We store it as-is.
    """
    try:
        # Validate it's JSON; store the parsed object
        payload = json.loads(payload_json)
    except Exception as e:
        return {"ok": False, "error": f"payload_json not valid JSON: {e}"}
    rec = BUS.put(topic=topic, key=key, payload=payload, producer=producer, run_id=run_id)
    return {"ok": True, "record": rec}

@function_tool
@trace_tool("bus_get")
def bus_get(topic: str, key: str) -> dict:
    """
    Returns payload_json: JSON string or null if missing.
    """
    val = BUS.get(topic=topic, key=key)
    return {"ok": val is not None, "payload_json": (json.dumps(val) if val is not None else None)}