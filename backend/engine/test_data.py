from backend.engine.data import MarketDataEngine


if __name__ == "__main__":

    engine = MarketDataEngine()

    market_data = engine.load_csv(
        "data/sample_eurusd_m15.csv",
        symbol="EURUSD",
        timeframe="M15",
    )

    result = engine.validate(market_data)

    print("========================================")
    print("       NEXA FUNDS AI")
    print("      MARKET DATA ENGINE")
    print("========================================")
    print()

    print("Symbol:", market_data.symbol)
    print("Timeframe:", market_data.timeframe)
    print("Rows:", result["rows"])
    print("Start:", result["start"])
    print("End:", result["end"])
    print("Valid:", result["valid"])
    print()

    print("Columns:")
    print(list(market_data.dataframe.columns))
    print()

    print("First candles:")
    print(market_data.dataframe.head())
    print()

    print("NEXA FUNDS AI MARKET DATA ENGINE OK")
