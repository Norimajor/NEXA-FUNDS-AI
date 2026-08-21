from backend.engine.instrument import InstrumentEngine, InstrumentSpec


def create_test_engine():

    engine = InstrumentEngine()

    engine.register(
        InstrumentSpec(
            symbol="EURUSD",
            asset_class="forex",
            digits=5,
            point=0.00001,
            tick_size=0.00001,
            tick_value=1.0,
            contract_size=100000,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            pip_size=0.0001,
            currency_base="EUR",
            currency_quote="USD",
        )
    )

    engine.register(
        InstrumentSpec(
            symbol="USDJPY",
            asset_class="forex",
            digits=3,
            point=0.001,
            tick_size=0.001,
            tick_value=1.0,
            contract_size=100000,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            pip_size=0.01,
            currency_base="USD",
            currency_quote="JPY",
        )
    )

    engine.register(
        InstrumentSpec(
            symbol="XAUUSD",
            asset_class="metal",
            digits=2,
            point=0.01,
            tick_size=0.01,
            tick_value=1.0,
            contract_size=100,
            volume_min=0.01,
            volume_max=100.0,
            volume_step=0.01,
            pip_size=None,
            currency_base="XAU",
            currency_quote="USD",
        )
    )

    return engine


if __name__ == "__main__":

    engine = create_test_engine()

    print("========================================")
    print("       NEXA FUNDS AI")
    print("   INSTRUMENT ENGINE TEST")
    print("========================================")
    print()

    print("Registered instruments:")
    print(engine.list_instruments())
    print()

    print("EURUSD:")
    print("10 pips =", engine.calculate_pip_distance("EURUSD", 10))
    print()

    print("USDJPY:")
    print("10 pips =", engine.calculate_pip_distance("USDJPY", 10))
    print()

    print("XAUUSD:")
    print("100 points =", engine.calculate_price_distance("XAUUSD", 100))
    print()

    print("NEXA FUNDS AI INSTRUMENT ENGINE OK")
