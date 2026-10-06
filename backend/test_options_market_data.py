import unittest
from datetime import datetime, timezone

from market_data.analytics import compute_iv_statistics
from market_data.models import OptionContract
from market_data.normalizer import normalize_deribit_contract, normalize_gold_contract
from market_data.options import OptionsSignalEngine


class OptionsMarketDataTests(unittest.TestCase):
    def test_normalize_deribit_contract(self):
        raw = {
            "instrument_name": "BTC-25DEC26-62000-C",
            "timestamp": 1763817600000,
            "underlying_price": 63625.5,
            "expiration": 1763923200000,
            "strike": 62000.0,
            "option_type": "call",
            "contract_symbol": "BTC-25DEC26-62000-C",
            "bid": 1425.0,
            "ask": 1465.0,
            "last_price": 1442.5,
            "mark_price": 1444.0,
            "volume": 125,
            "open_interest": 300,
            "implied_volatility": 0.368,
            "bid_iv": 0.36,
            "ask_iv": 0.374,
            "delta": 0.53,
            "gamma": 0.0012,
            "theta": -0.37,
            "vega": 0.26,
            "moneyness": 1.04,
        }

        contract = normalize_deribit_contract(raw)

        self.assertIsInstance(contract, OptionContract)
        self.assertEqual(contract.asset, "BTC")
        self.assertEqual(contract.provider, "deribit")
        self.assertEqual(contract.option_type, "call")
        self.assertEqual(contract.contract_symbol, "BTC-25DEC26-62000-C")
        self.assertEqual(contract.strike, 62000.0)
        self.assertEqual(contract.bid, 1425.0)
        self.assertEqual(contract.ask_iv, 0.374)
        self.assertIsNotNone(contract.timestamp)
        self.assertIsNotNone(contract.days_to_expiration)

    def test_normalize_gold_contract(self):
        raw = {
            "symbol": "COMEX:G2W260909C4450",
            "strike": 4450.0,
            "option_type": "call",
            "expiration": "20260909",
            "underlying_price": 4436.75,
            "volume": 146,
            "open_interest": 521,
            "iv": 0.274,
            "bid_iv": 0.268,
            "ask_iv": 0.281,
            "delta": 0.253,
            "gamma": 0.0068,
            "theta": -13.85,
            "vega": 0.17,
            "bid": 6.95,
            "ask": 7.18,
            "last_price": 7.08,
            "mark_price": 7.12,
            "moneyness": 0.9992,
        }

        contract = normalize_gold_contract(raw)

        self.assertEqual(contract.asset, "XAU")
        self.assertEqual(contract.provider, "tvdataoption")
        self.assertEqual(contract.option_type, "call")
        self.assertEqual(contract.expiration, "20260909")
        self.assertEqual(contract.strike, 4450.0)
        self.assertEqual(contract.underlying_price, 4436.75)
        self.assertEqual(contract.iv, 0.274)
        self.assertEqual(contract.bid_iv, 0.268)
        self.assertEqual(contract.ask_iv, 0.281)

    def test_compute_iv_statistics(self):
        contracts = [
            OptionContract(
                asset="BTC",
                provider="deribit",
                timestamp=datetime.now(timezone.utc),
                underlying_price=60000.0,
                expiration="20261101",
                days_to_expiration=30,
                strike=60000.0,
                option_type="call",
                contract_symbol="BTC-01NOV26-60000-C",
                bid=None,
                ask=None,
                last_price=None,
                mark_price=None,
                volume=None,
                open_interest=None,
                implied_volatility=0.32,
                bid_iv=0.3,
                ask_iv=0.34,
                delta=0.51,
                gamma=0.001,
                theta=-0.3,
                vega=0.2,
                moneyness=None,
            ),
            OptionContract(
                asset="BTC",
                provider="deribit",
                timestamp=datetime.now(timezone.utc),
                underlying_price=60000.0,
                expiration="20261101",
                days_to_expiration=30,
                strike=61000.0,
                option_type="put",
                contract_symbol="BTC-01NOV26-61000-P",
                bid=None,
                ask=None,
                last_price=None,
                mark_price=None,
                volume=None,
                open_interest=None,
                implied_volatility=0.38,
                bid_iv=0.36,
                ask_iv=0.4,
                delta=-0.48,
                gamma=0.001,
                theta=-0.35,
                vega=0.22,
                moneyness=None,
            ),
        ]

        stats = compute_iv_statistics(contracts)

        self.assertEqual(stats["atm_iv"], 0.32)
        self.assertEqual(stats["iv_by_strike"][60000.0], 0.32)
        self.assertEqual(stats["iv_by_expiration"]["20261101"], 0.35)

    def test_options_signal_engine_generates_deterministic_signal(self):
        contracts = [
            OptionContract(
                asset="BTC",
                provider="deribit",
                timestamp=datetime.now(timezone.utc),
                underlying_price=60000.0,
                expiration="20261101",
                days_to_expiration=30,
                strike=60000.0,
                option_type="call",
                contract_symbol="BTC-01NOV26-60000-C",
                bid=None,
                ask=None,
                last_price=2000.0,
                mark_price=2000.0,
                volume=100,
                open_interest=1500,
                implied_volatility=None,
                bid_iv=0.25,
                ask_iv=0.27,
                delta=0.62,
                gamma=0.002,
                theta=-0.45,
                vega=0.18,
                moneyness=None,
            ),
            OptionContract(
                asset="BTC",
                provider="deribit",
                timestamp=datetime.now(timezone.utc),
                underlying_price=60000.0,
                expiration="20261101",
                days_to_expiration=30,
                strike=61000.0,
                option_type="put",
                contract_symbol="BTC-01NOV26-61000-P",
                bid=None,
                ask=None,
                last_price=1800.0,
                mark_price=1800.0,
                volume=80,
                open_interest=1300,
                implied_volatility=None,
                bid_iv=0.28,
                ask_iv=0.30,
                delta=-0.43,
                gamma=0.0015,
                theta=-0.4,
                vega=0.15,
                moneyness=None,
            ),
            OptionContract(
                asset="BTC",
                provider="deribit",
                timestamp=datetime.now(timezone.utc),
                underlying_price=60000.0,
                expiration="20261101",
                days_to_expiration=30,
                strike=59000.0,
                option_type="call",
                contract_symbol="BTC-01NOV26-59000-C",
                bid=None,
                ask=None,
                last_price=1700.0,
                mark_price=1700.0,
                volume=70,
                open_interest=900,
                implied_volatility=None,
                bid_iv=0.24,
                ask_iv=0.26,
                delta=0.54,
                gamma=0.0017,
                theta=-0.38,
                vega=0.13,
                moneyness=None,
            ),
        ]

        engine = OptionsSignalEngine()
        signal = engine.generate(contracts, asset="BTC", reference_price=60000.0)
        self.assertIn(signal["signal"], {"BUY", "SELL", "HOLD"})
        self.assertGreaterEqual(signal["confidence"], 0.0)
        self.assertLessEqual(signal["confidence"], 1.0)
        self.assertIn("data_quality", signal)
        self.assertIn("reasons", signal)


if __name__ == "__main__":
    unittest.main()
