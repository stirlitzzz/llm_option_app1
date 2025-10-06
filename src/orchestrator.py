# src/orchestrator.py
import os, logging
from dotenv import load_dotenv, find_dotenv
from datetime import date
load_dotenv(find_dotenv(), override=True)

# quiet httpx unless you want network noise
logging.getLogger("httpx").setLevel(logging.WARNING)

from agents import Runner
from agent_strategy import strategy_agent
from agent_pricer import pricer_agent
from agent_coach import coach_agent
from tools_parser import parse_and_store_mvp, parse_and_store_mvp_impl, normalize_and_store_legs_impl

HELP = (
    "Commands:\n"
    "  strategy translator: <free text>\n"
    "  pricer: <key>\n"
    "Examples:\n"
    "  strategy translator: i want to trade 500 AAPL jan26 250 300 collars\n"
    "  pricer: AAPL-2026-01-collar\n"
)

def route(user_text: str):
    ref_date=date.today()
    txt = user_text.strip()
    if txt.lower().startswith("strategy translator:"):
        payload = txt.split(":", 1)[1].strip()
        res = Runner.run_sync(strategy_agent, payload)
        return res.final_output
    if txt.lower().startswith("pricer:"):
        key = txt.split(":", 1)[1].strip()
        res = Runner.run_sync(pricer_agent, key)
        return res.final_output
    if txt.lower().startswith("parse strategy:"):
        payload = txt.split(":", 1)[1].strip()
        out = parse_and_store_mvp_impl(text=payload)  # direct Python call
        out2=normalize_and_store_legs_impl(key=out["key"], ref_year=ref_date.year, ref_month=ref_date.month, ref_day=ref_date.day)
        #res = Runner.run_sync(out, payload
        print(f"out2: {out2}")
        print(f"out: {out}")
        return out2
    return Runner.run_sync(coach_agent, txt).final_output
    return HELP

if __name__ == "__main__":
    # demo flow
    out1 = route("parse strategy: translate this request: i want to trade 500 AAPL jan26 250 300 collars")
    print("\n🔑 key:", out1)
    out2 = route(f"pricer: {out1}")
    print("\n📊 result:\n", out2)
    out3 = route(f"coach: {out2}")
    print("\n📊 result:\n", out3)
