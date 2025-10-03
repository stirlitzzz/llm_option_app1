This is a great step — you’re moving from “half-baked stubs” to a rules of the road document for option strategy parsing. Let me take what you wrote and structure it into something you (and eventually the agents or a Python parser) can reference. I’ll also point out conflicts/ambiguities and add suggestions.

⸻

Draft: Option Strategy Parsing Rules

Overall Input Order

After any “free text” intro, strategy inputs always appear in this order:
	1.	Quantity
	2.	Ticker
	3.	Maturities
	4.	Strikes
	5.	Optional Ratio
	6.	Structure

⸻

1. Quantity
	•	One integer.
	•	May include comma separators (e.g. 1000 or 1,000).
	•	Quantity scales the default ratios for all legs of the strategy.

⸻

2. Stock Ticker
	•	Standard ticker, uppercase letters (1–5 characters typical).
	•	Example: AAPL, TSLA, MSFT.

⸻

3. Maturities
	•	Can be:
	•	MMMYY (Jan26)
	•	MMMYYYY (Jan2026)
	•	MMM DD (Jan 26)
	•	Month YYYY (January 2026)
	•	Month DD (January 26)
	•	Exact date (MM/DD/YYYY or MM/DD/YY)
	•	Multiple maturities are allowed.
	•	Application rule: If fewer maturities are provided than option legs, the last one is repeated.

⸻

4. Strikes
	•	Can be integers or floats.
	•	Separated by space or /.
	•	Multiple strikes are allowed.
	•	Application rule: If fewer strikes are provided than option legs, the last one is repeated.

⸻

5. Optional Ratio
	•	Format: NxM, case-insensitive (1x2 or 1X2).
	•	Multiplies the default quantity ratio for each leg.
	•	Example:
	•	Call spread default = +1C, -1C
	•	With 1x2 → +1C, -2C
	•	Application rule: If ratio length < number of legs, apply last ratio to remaining legs.

⸻

6. Structure

Recognized keywords and aliases:
	•	Straddle / Strangle / Str
	•	Treated equivalently.
	•	If 1 strike given → straddle.
	•	If 2+ strikes given → strangle.
	•	Risk Reversal / RR
	•	Same as collar, but short put and long call.
	•	Collar
	•	Default long put, short call.
	•	Call Spread / CS
	•	Two calls, typically buy lower K, sell higher K.
	•	Put Spread / PS
	•	Two puts, typically buy higher K, sell lower K.
	•	String of C/P characters (e.g. PPC)
	•	Each character corresponds to one leg.
	•	Default ratio = 1 per leg (scaled by input quantity and optional ratio).
	•	Example: PPC with 1x2 → Put(1), Put(2), Call(2).
	•	Put Spread Collar / PS Collar
	•	Collar variant where the put side is a spread.
	•	Call Spread Collar / CS Collar
	•	Collar variant where the call side is a spread.

⸻

Application Rules
	•	Mismatch filling:
	•	If maturities, strikes, or ratios have fewer entries than legs, propagate the last one forward.
	•	Default sides (BUY/SELL):
	•	Collar = long put, short call.
	•	Risk reversal = short put, long call.
	•	Spreads = buy one, sell one (by convention: debit spread = buy closer-to-money leg).
	•	C/P string = default to BUY unless overridden by ratio sign or explicit side.
	•	Default expiry day:
	•	If only month/year given, assume 3rd Friday of that month.

⸻

1. Conflicts / Ambiguities
	•	“Straddle” vs “Strangle” vs “Str”:
You’re treating them the same, but in trading parlance they differ (ATM vs OTM). Suggest: keep them separate in parsing but allow “str” abbreviation → disambiguate based on strike count.
	•	Ratios with C/P strings:
Example PPC with 1x2 → should ratio apply as [1,2,2]? That’s what you wrote, but it’s non-standard. Clarify that convention.
	•	Spread direction (call spread vs put spread):
Does the parser infer debit vs credit automatically, or just assign “buy first, sell second”? Might need a consistent rule.
	•	Maturity formats:
Jan 26 → could be interpreted as Jan 2026 (month+year) or Jan 26th (exact date). Ambiguity exists. Suggest: if a day is present and ≤31, treat it as a date. Otherwise treat as year.
	•	Risk Reversal vs Collar:
Risk reversal = short put/long call. Collar = long put/short call. Both sometimes mean “buy/sell protection around stock.” Clarify if you want stock leg included or not.

⸻

2. Suggestions
	•	Normalize all inputs immediately:
	•	Strip commas in numbers, uppercase tickers, standardize expiries → ISO date (YYYY-MM-DD).
	•	Schema-first: define an internal JSON format (legs with cp, strike, expiry, side, qty). Every parser rule should map into it.
	•	Keep a “structure dictionary”: map aliases (cs, call spread) to canonical form.
	•	Explicit vs inferred: mark inferred sides/ratios with "side":"AUTO" so downstream can choose to ask the user for confirmation.
	•	Versioning: strategies evolve; add "parser_version": "0.1" to payloads so you know what rules created them.
	•	Extensibility: allow a "notes" field where leftover text is stored (good for debugging).

⸻

3. Reformatted Document (Ready-to-Use)

I’d put this in a STRATEGY_RULES.md so both humans and agents can reference it.

⸻

Would you like me to also sketch the canonical JSON schema for one strategy (all normalized fields), so you can test both your Python parser and the LLM parser against the same contract?
