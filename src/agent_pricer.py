# src/agent_pricer.py
from agents import Agent
from tools_bus import bus_get, bus_put
from agents_wireup import resolve_series, load_quotes, load_curve, build_div_forecast, price_with_ql, present_rows
import json

pricer_agent = Agent(
    name="Pricer",
    model="gpt-4.1-mini",
    #temperature=0.0,
    instructions=(
        "Given a key, price the strategy stored on the bus.\n"
        "1) bus_get('spec', key) → payload_json; parse to spec\n"
        "2) resolve_series(spec) → ids\n"
        "3) load_quotes(ids) → quotes\n"
        "4) load_curve('curves/latest.csv') and build_div_forecast(spec.symbol, [ids[0].expiration])\n"
        "5) price_with_ql(legs=ids, curve=curve, divs=divs) → priced\n"
        "6) present_rows(priced, quotes) → ui\n"
        "7) bus_put('pricing', key, payload_json=JSON.stringify({'ui': ui, 'priced': priced}), producer='pricer')\n"
        "Return a short textual summary and the key."
    ),
    tools=[bus_get, resolve_series, load_quotes, load_curve, build_div_forecast, price_with_ql, present_rows, bus_put],
)
