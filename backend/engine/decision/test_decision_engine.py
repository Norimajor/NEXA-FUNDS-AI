from backend.engine.decision.decision_engine import DecisionEngine


print("=" * 40)
print("       NEXA FUNDS AI")
print("       DECISION ENGINE TEST")
print("=" * 40)


# ---------------------------------------------------------
# BULLISH MARKET SCENARIO
# ---------------------------------------------------------

market = {
    "close": 1.1050,

    # Regime
    "TREND_REGIME": "BULL",
    "MARKET_CONDITION": "TRENDING",
    "VOLATILITY_REGIME": "NORMAL",
    "MARKET_REGIME": "BULL_TRENDING_NORMAL",

    # Structure
    "HIGH_STRUCTURE": "HH",
    "LOW_STRUCTURE": "HL",

    # BOS / CHOCH
    "BULLISH_BOS": True,
    "BEARISH_BOS": False,
    "BULLISH_CHOCH": False,
    "BEARISH_CHOCH": False,

    # Liquidity
    "SELL_SIDE_SWEEP": True,
    "BUY_SIDE_SWEEP": False,

    # FVG
    "BULLISH_FVG": True,
    "BEARISH_FVG": False,

    # Order block
    "BULLISH_ORDER_BLOCK": True,
    "BEARISH_ORDER_BLOCK": False,

    # Premium / discount
    "IN_DISCOUNT": True,
    "IN_PREMIUM": False,

    # Momentum
    "RSI_14": 62,
    "ADX_14": 29,

    # Volume
    "RELATIVE_VOLUME": 1.42,

    # ATR
    "ATR": 0.0010,
}


# ---------------------------------------------------------
# ENGINE
# ---------------------------------------------------------

engine = DecisionEngine(
    minimum_score=60,
    strong_score=80,
    risk_reward=2.0,
)


# ---------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------

result = engine.evaluate(market)


# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print("\nDECISION")
print("-" * 40)

print(f"Signal: {result.signal}")
print(f"Confidence: {result.confidence}%")

print(f"Bullish score: {result.bullish_score}")
print(f"Bearish score: {result.bearish_score}")

print("\nMARKET")
print("-" * 40)

print(f"Trend: {result.trend}")
print(f"Market condition: {result.market_condition}")
print(f"Volatility: {result.volatility_regime}")
print(f"Regime: {result.market_regime}")


print("\nTRADE PLAN")
print("-" * 40)

print(f"Entry: {result.entry}")
print(f"Stop Loss: {result.stop_loss}")
print(f"Take Profit: {result.take_profit}")
print(f"Risk/Reward: {result.risk_reward}")


print("\nREASONS")
print("-" * 40)

for reason in result.reasons:
    print(f"✓ {reason}")


print("\nEXPECTED")
print("-" * 40)

print("Signal should be BUY or STRONG_BUY")
print("Bullish evidence should dominate")
print("Entry, SL and TP should be generated")


print("\nNEXA FUNDS AI DECISION ENGINE OK")