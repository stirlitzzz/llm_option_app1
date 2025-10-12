# orchestrator.py
from agents import Agent, Runner

from agents_wireup import (
    parse_intent, resolve_series, load_quotes, load_curve, build_div_forecast, price_with_ql, present_rows
)

from dotenv import load_dotenv, find_dotenv
import os
import logging
logging.basicConfig(level=logging.INFO)

# SDK internals (tool planning, arguments)
logging.getLogger("agents").setLevel(logging.DEBUG)
logging.getLogger("agents.tool").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.WARNING)  # or DEBUG if you want payload sizes
logging.getLogger("httpx").setLevel(logging.INFO)      # you already have this
#assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY"
load_dotenv(find_dotenv(), override=True)
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in .env file or environment.")

orchestrator = Agent(
    name="Options-Orchestrator",
    model="gpt-4.1-nano",   # 👈 explicitly set

    instructions=(
        "You are a trading workflow agent. "
        "Given user text, call tools in this order:\n"
        "1) parse_intent(text) → spec in normalized.spec\n"
        "2) resolve_series(spec) → dataset IDs\n"
        "3) load_quotes(dataset) → add bid/ask/iv/spot\n"
        "4) load_curve('curves/latest.csv') and build_div_forecast(symbol, [expiration])\n"
        "5) price_with_ql(legs=dataset, curve_rows, divs_by_exp) → theo+greeks\n"
        "6) present_rows(priced_legs) → final table\n"
        "If any step returns ok=false or issues with 'blocker', stop and surface a concise message."
    ),
    tools=[parse_intent, resolve_series, load_quotes, load_curve, build_div_forecast, price_with_ql, present_rows],
)

# Run it:
if __name__ == "__main__":
    user_text = "price an AAPL Jan26 250/300 collar"
    result = Runner.run_sync(orchestrator, user_text)
    print(result.final_output)         # the presenter’s compact rows
    # You can also inspect tool traces via result.spans / tracing UI.
