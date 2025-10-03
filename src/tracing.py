# src/tracing.py
import time, json, functools

def _safe_json(x, maxlen=1200):
    try:
        s = json.dumps(x, default=str)
    except Exception:
        s = str(x)
    return (s[:maxlen] + "…") if len(s) > maxlen else s

def trace_tool(name: str):
    """Console tracer. IMPORTANT: put above @function_tool so it wraps the real function."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            t0 = time.time()
            print(f"\n▶️  TOOL START: {name}")
            if args:   print("   args   :", _safe_json(args))
            if kwargs: print("   kwargs :", _safe_json(kwargs))
            out = fn(*args, **kwargs)
            ms = (time.time() - t0) * 1000
            if isinstance(out, dict):
                print(f"✅ TOOL END  : {name} ({ms:.1f} ms) keys={list(out.keys())}")
                if "issues" in out and out["issues"]:
                    print("   issues :", _safe_json(out["issues"]))
                if "dataset" in out:
                    try: print("   dataset.len :", len(out["dataset"]))
                    except Exception: pass
            else:
                print(f"✅ TOOL END  : {name} ({ms:.1f} ms) type={type(out).__name__}")
            return out
        return wrapped
    return deco
