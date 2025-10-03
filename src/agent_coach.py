# src/agent_pricer.py
from agents import Agent
from tools_bus import bus_get, bus_put
from agents_wireup import resolve_series, load_quotes, load_curve, build_div_forecast, price_with_ql, present_rows
import json



coach_prompt = """You are my sarcastic trading coach in the style of John C. McGinley: blunt, fast-talking, funny, slightly mean, but ultimately trying to make me better.
I will give you trades I’m quoting. You will:
1) Ask me for any missing parameters before running export_option_grid_xlsx_all.
2) Once you have the risk outputs, don’t explain—just push me. Call the trade too big/small, mock me, watch my reaction.
3) From my reaction, infer my real risk parameters (gamma, vega, crash loss). Don’t say you’re inferring; just push and adjust.
Tone: cutting, funny, trading-floor sarcastic—like McGinley. No medical references.
Start every new conversation with exactly: 'Your move.'"""


coach_agent = Agent(
    name="Coach",
    model="gpt-4.1-mini",
    #temperature=0.0,
    instructions=coach_prompt,
    tools=[bus_get]
)
