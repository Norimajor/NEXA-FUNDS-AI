import unittest
from datetime import datetime, timezone

from market_data.barchart_options import (
    BarchartOptionsProvider,
    BarchartRawOption,
    parse_call_put,
    parse_expiration,
    parse_numeric,
    parse_option_type,
    parse_strike,
)


class BarchartScraperTests(unittest.TestCase):
    def test_parse_expiration_accepts_common_formats(self):
        self.assertEqual(parse_expiration("2026-09-30"), "2026-09-30")
        self.assertEqual(parse_expiration("9/30/2026"), "2026-09-30")
        self.assertEqual(parse_expiration("20260930"), "2026-09-30")
        self.assertIsNone(parse_expiration("bad-date"))

    def test_parse_numeric_and_strike(self):
        self.assertEqual(parse_numeric("1,234.50"), 1234.5)
        self.assertEqual(parse_numeric("--"), None)
        self.assertEqual(parse_strike("2450.0"), 2450.0)
        self.assertIsNone(parse_strike("N/A"))

    def test_option_type_detection(self):
        self.assertEqual(parse_option_type("Call"), "call")
        self.assertEqual(parse_option_type("P"), "put")
        self.assertEqual(parse_option_type("C"), "call")
        self.assertIsNone(parse_option_type("Other"))

    def test_parse_call_put_from_symbol(self):
        self.assertEqual(parse_call_put("COMEX:G2U260930C2450"), "call")
        self.assertEqual(parse_call_put("COMEX:G2U260930P2450"), "put")

    def test_provider_normalizes_fixture_rows(self):
        provider = BarchartOptionsProvider()
        fixture = [
            {
                "symbol": "COMEX:G2U260930C2450",
                "strike": "2450.0",
                "expiration": "2026-09-30",
                "option_type": "Call",
                "bid": "8.25",
                "ask": "8.75",
                "last": "8.50",
                "open_interest": "521",
                "volume": "146",
                "implied_volatility": "0.274",
                "delta": "0.253",
                "gamma": "0.0068",
                "theta": "-13.85",
                "vega": "0.17",
                "underlying_price": "2448.20",
            },
            {
                "symbol": "COMEX:G2U260930P2450",
                "strike": "2450.0",
                "expiration": "2026-09-30",
                "option_type": "Put",
                "bid": "8.10",
                "ask": "8.65",
                "last": "8.35",
                "open_interest": "495",
                "volume": "130",
                "implied_volatility": "0.289",
                "delta": "-0.241",
                "gamma": "0.0061",
                "theta": "-13.10",
                "vega": "0.16",
                "underlying_price": "2448.20",
            },
        ]

        contracts = provider._normalize_rows(fixture, asset="GOLD", proxy_for="XAUUSD")
        self.assertEqual(len(contracts), 2)
        self.assertEqual(contracts[0].asset, "GOLD")
        self.assertEqual(contracts[0].provider, "barchart_scrape")
        self.assertEqual(contracts[0].option_type, "call")
        self.assertEqual(contracts[0].strike, 2450.0)
        self.assertIsNotNone(contracts[0].timestamp)

    def test_raw_option_model_tracks_source(self):
        raw = BarchartRawOption(
            raw_symbol="COMEX:G2U260930C2450",
            source_url="https://www.barchart.com/futures/options/GC",
            scraped_at=datetime.now(timezone.utc),
            raw={"symbol": "COMEX:G2U260930C2450", "strike": "2450.0"},
        )
        self.assertEqual(raw.source, "barchart_scrape")
        self.assertEqual(raw.raw_symbol, "COMEX:G2U260930C2450")

    def test_zero_rows_are_reported_as_failed(self):
        provider = BarchartOptionsProvider(cache_seconds=0.0)
        status = provider._status_result(success=False, symbol="GC", records=0, fields_available=[], fields_missing=[], error="NO_OPTION_ROWS_FOUND")
        self.assertFalse(status["success"])
        self.assertEqual(status["records"], 0)


if __name__ == "__main__":
    unittest.main()
