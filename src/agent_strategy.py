# src/agent_strategy.py
from agents import Agent
from tools_bus import bus_put
from agents_wireup import translate_strategy
from tools_parse import parse_strategy_python
from strategy_fast_parse import parse_strategy_mvp
import json

from agents import function_tool
from tracing import trace_tool

strategy_agent = Agent(
    name="Strategy-Translator",
    model="gpt-4.1-mini",
    #temperature=0.0,
    instructions=(
        "Translate the user's strategy request to a strict spec.\n"
        "1) Call translate_strategy(text) → spec\n"
        "2) Build key '<SYMBOL>-<YYYY>-<MM>-<structure>'\n"
        "3) Call bus_put(topic='spec', key=key, payload_json=JSON.stringify(spec), producer='strategy')\n"
        "4) Reply ONLY with the key."
    ),
    tools=[parse_strategy_python, bus_put],
)



