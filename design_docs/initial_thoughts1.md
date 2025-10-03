Trading Coach — Concept Spec (v0)

Goal: A chat-first “junior trader” that turns a free-text quote into files + crisp risk commentary, one push at a time.
First wow: Link to EOD files and 3 bullets in ≤5s. Then a single, concrete next knob.

⸻

1) North Star

User: “I’m quoting 3mm NVDA 2y 85/125 collar @ 40/32 vols.”
Agent replies (one message, link-first):
	•	Links: EOD structure CSV + vol curve PNG
	•	Bullets (value-only): structure mid/bid/ask; ATM vol & slope vs curve; basis to listed
	•	Ask: 1 clarification (e.g., “Confirm r=4.3%?”)
	•	Next: risk grid → link + 3 bullets (dollar-gamma, vega, worst down-jump) + one knob suggestion

⸻

2) Architecture (backend-first)
	•	Agents SDK on the server (single agent or light multi-agent), tools = plain server functions (no MCP to start).
	•	FastAPI exposes a single endpoint now (POST /agent/step) and a WebSocket later (/ws).
	•	Files saved locally (or S3); reply with public links only.
	•	UI contract is a UI Patch (see §5): front end just applies patches.

⸻

3) Roles (lightweight team)
	•	Coordinator – decides the step: Validate → EOD → Price Structure → Risk Grid → Coach.
	•	Validator – normalizes inputs; asks one missing/ambiguous item per turn.
	•	Coach/Explainer – formats 3 bullets + 1 line of snark (no coordinates, $mm).
	•	Goal/Knowledge – keeps the objective: understand risk params; hints push order (gamma → vega → crash → carry) with short glossary.

Start with one agent and encode these in the system prompt + tool descriptions; split later if helpful.

⸻

4) Tool Functions (server-side, deterministic)

Each tool returns strict JSON and file links; no prose.

4.1 validate_inputs

In: partial { ticker, spot?, r?, q?, contract_size?, legs[]: { cp, strike, vol, qty, ttm } }
Out:

{
  "ok": false,
  "missing": ["r"],
  "ambiguous": [{"field":"r","message":"r looks high; confirm units","candidates":[0.4,0.04,0.004]}],
  "normalized": { "ticker":"NVDA", "spot":170, "r":0.043, "q":0.0, "contract_size":1, "legs":[...] }
}

4.2 load_eod_chain

Fetch yesterday’s listed chain (initially from ivol snapshot).
In: { ticker, date? }
Files: chain.csv, curve.png (ATM vs tenor or smile + marked strikes)
Out:

{
  "atm_vol": 0.33,
  "slope": -0.08,
  "curve_stats": { "tenors":[0.25,0.5,1,2], "atm":[0.29,0.31,0.32,0.33] },
  "files": { "csv":"https://.../chain.csv", "png":"https://.../curve.png" }
}

4.3 price_structure_eod

Price the proposed structure vs listed markets.
In: normalized inputs + chain.csv
Files: structure_eod.csv, optionally overlay.png (strikes on smile)
Out:

{
  "mid": 1.23, "bid": 1.10, "ask": 1.36, "basis_vs_listed": -0.04,
  "per_leg": [{ "label":"85% put", "mid":..., "iv":..., "greeks":{...} }, ...],
  "files": { "csv":"https://.../structure_eod.csv", "png":"https://.../overlay.png" }
}

4.4 export_risk_grid

Your existing engine.
In: normalized inputs
Files: grid_all_*.xlsx (+ optional heatmap.png later)
Out (value-only summary in $mm):

{
  "download_url": "https://.../grid_all_*.xlsx",
  "summary": {
    "dollar_gamma_per_1pct": { "min": -13.6e6, "max": 24.9e6 },
    "vega_per_volpt":        { "min": -0.96e6, "max": 1.55e6 },
    "jump_pnl_dn_most_negative": { "value": -433.5e6, "jump_pct_used": -1.0 }
  },
  "bullets": [
    "Dollar gamma (±1%): -13.60mm ↔ 24.90mm",
    "Vega per 1 vol pt:  -0.96mm ↔ 1.55mm",
    "Worst DN jump: -433.50mm (jump -100%)"
  ],
  "next": { "knob": "sigma", "leg_idx": 1, "values": [0.27,0.30,0.32,0.34,0.37] }
}

4.5 (later) sweep_one

In: { knob:"sigma"|"strike"|"ttm"|"qty", leg_idx, values[], inputs }
Files: sweep_<knob>.csv, sweep_<knob>.png
Out: { "rows":[{"value":0.30,"price":...,"vega":...},...], "best": {"value":0.34}, "files": {...} }

⸻

5) UI Contract (patch)

The backend produces a UI Patch the frontend merges into its state. Same shape over HTTP or WS.

{
  "sessionId": "abc123",
  "feedAppend": [{ "who":"coach", "lines": ["Your move."] }],
  "files": [{ "label":"Chain CSV", "url":"https://.../chain.csv" }, { "label":"Vol curve", "url":"https://.../curve.png" }],
  "risk": {
    "download_url": "https://.../grid_all.xlsx",
    "bullets": ["Dollar gamma: -13.60mm ↔ 24.90mm", "Vega: -0.96mm ↔ 1.55mm", "Worst DN jump: -433.50mm"],
    "summary": { "...": "..." }
  },
  "knob": { "name":"sigma","legIdx":1,"value":0.32,"min":0.27,"max":0.37,"step":0.01 },
  "charts": { "sweep_price": [{ "x":0.27, "y":12.3 }, { "x":0.30, "y":12.9 }] },
  "rev": 9
}

WS event order per turn: chat → file → bullets → knob → chart → done

⸻

6) Conversation Flow (first phase)
	1.	Validate
	•	If missing/ambiguous → ask one thing (e.g., “Confirm r=4.3% annualized?”).
	2.	EOD snapshot
	•	load_eod_chain → price_structure_eod
	•	Reply: links (chain, overlay), 3 bullets (mid/bid/ask, ATM, slope/basis), 1 ask if needed.
	3.	Risk grid (on confirm)
	•	export_risk_grid → reply: link, 3 bullets, next knob.
	4.	Knob (on accept)
	•	Optional sweep_one (line chart); else re-export with the applied change.

⸻

7) State (server)

{
  "inputs": { "ticker":"NVDA","spot":170,"r":0.043,"q":0.0,"contract_size":1,"legs":[...] },
  "eod":     { "files": {...}, "atm_vol": 0.33, "slope": -0.08 },
  "structure": { "mid": 1.23, "bid": 1.10, "ask": 1.36, "basis": -0.04, "files": {...} },
  "risk":    { "files": {...}, "summary": {...}, "last_next": {...} },
  "history": [ { "time": "...", "step":"export", "notes":"..." } ]
}

Persist per sessionId (Redis/SQLite). Don’t store XLSX blobs—store links.

⸻

8) Prompting Rules (concise)
	•	System tone: link first, 3 bullets ($mm, 2 decimals, value-only), then 1 snark.
	•	Ask policy: exactly one clarification at a time.
	•	Push order: gamma → vega → crash → carry (unless user asks otherwise).
	•	Tool usage: deterministic tools do the heavy lifting; agent formats and nudges.

⸻

9) Milestones

M1 — EOD MVP (no charts)
	•	/agent/step returns: chain link, overlay link, 3 bullets, 1 ask.

M2 — Risk MVP
	•	Run export_risk_grid and include link + 3 bullets + next.

M3 — Knob & Chart
	•	Add sweep_one; return CSV + simple sweep chart points.

M4 — WS Streaming
	•	Stream chat → file → bullets → knob → chart → done.

M5 — Parse & Memory
	•	Replace seed legs with validator; persist session state.

⸻

10) Guardrails
	•	Only one knob active per turn; max one sweep per turn.
	•	Strict JSON from tools; no prose.
	•	Always link first before commentary.
	•	Cap tool calls/time per turn (budget).
	•	Log tool name, args, outputs per turn for auditability.

⸻

11) Useful From Day One
	•	EOD structure basis vs listed, with files you can share.
	•	Risk grid link + 3 bullets that a trader can act on.
	•	A single, concrete next knob (e.g., “bump long-call vol ±5pts”).

⸻

This spec keeps the UI contract stable while you iterate on the backend engine (Agents SDK now, LangGraph or custom FSM later). Build the content first; rendering can follow the same uiPatch.
