# NexaFunds Analysis AI — Learning Backend

Real backend for NexaFunds: historical learning + supply/demand + smart channel detection + market regime + risk-aware signal generation.

## Install/run
`py -m venv .venv`
`.venv\\Scripts\\activate`
`pip install -r requirements.txt`
`py main.py`

## Train from MT5 history
Export chronological OHLCV CSV with columns `timestamp,open,high,low,close,volume`, then:
`py train_model.py your_history.csv --version model_001`

The classifier trains on causal indicator, confirmed-swing, repeated-level, candle-rejection, and liquidity-sweep features. Direction labels require the target barrier to be reached before the opposing stop; same-candle barrier conflicts are marked no-edge. Chronological holdout and walk-forward validation purge the 12-candle label horizon, and model promotion additionally requires minimum out-of-sample precision, recall, and support for both reversal directions. Failed candidates do not replace the active model. These checks reduce leakage and weak-model promotion but do not guarantee profitable trading.

The trained H1/M15 pivot-reversal candidates were force-promoted despite failing validation gates; their recorded metrics and failed gates are included with the model artifacts and registry. This is an explicit operator override, not evidence that either model is reliable or profitable. In particular, reversal precision is weak, so do not treat these predictions as trustworthy live-trading signals. The predictor maps the expanded live feature set onto older model feature schemas so model loading remains backward-compatible; old models do not gain the new reversal features without being retrained.

H1 and M15 trend bias is predicted by a separately trained three-class ML model using the same causal market-feature pipeline as the setup classifier. Its historical targets classify a 50-bar ATR-normalized price-path regression as `BUY`, `SELL`, or `RANGE`; the API does not manufacture a `NONE` trend by combining EMA votes. `RANGE` is an explicit model class and means the observed path does not meet either directional label threshold. The trend-only training option evaluates a chronological holdout and saves an independent artifact without replacing the setup classifier:

`py train_model.py backend/data/XAUUSD_H1.csv --trend-only --version xauusd_h1_trend_20261008`

Set `NEXA_H1_TREND_MODEL_PATH` to that artifact when deploying a differently named model. M15 uses the equivalent `NEXA_M15_TREND_MODEL_PATH`. Trend model quality is reported on the chronological holdout and must be inspected; it is not a guarantee of accurate live direction. The consensus API waits for a directional M15 setup before planning an entry. A counter-bias M15 trade is allowed only when the M15 model classifies it as a reversal or failed breakout; disagreement without one of those learned setup classes remains `WAIT`. A planned trade still waits for its structural entry zone to be touched, and the EA also checks that the live quote remains inside the zone before opening a position. Buy/sell probability totals remain in the API for backwards compatibility but are not used to choose the trade direction or shown as the EA's directional decision.

The EA's six-timeframe consensus request uses H4 for overall context and H1 for the mandatory current-trend direction. H1 must predict BUY or SELL at or above `NEXA_TREND_MIN_CONFIDENCE` (default 0.55); H4 disagreement is informational and does not block an otherwise valid trade. H4 agreement selects the more conservative target profile. M1 and M5 setup models must agree with H1 for a scalp; M15 and M30 setup models must agree with H1 for a swing entry. The request can select `AUTO`, `SCALP`, or `SWING`; AUTO prefers a confirmed swing and otherwise uses a confirmed scalp. The service waits when an entry model is missing, confidence is below the configured threshold, H1 has no usable direction, the paired entry models disagree, or the structural entry zone has not been touched. H4/H1 default to the pooled trend artifact at `models/pooled_all_timeframe_trend_20261017.joblib`, configurable with `NEXA_SHARED_TREND_MODEL_PATH` (or the per-timeframe H4/H1 overrides).

The six-timeframe route expects entry artifacts at `models/m1_entry_20261017.joblib`, `models/m5_entry_20261017.joblib`, `models/m15_entry_20261017.joblib`, and `models/m30_entry_20261017.joblib`, configurable with `NEXA_M1_ENTRY_MODEL_PATH`, `NEXA_M5_ENTRY_MODEL_PATH`, `NEXA_M15_ENTRY_MODEL_PATH`, and `NEXA_M30_ENTRY_MODEL_PATH`. Train a timeframe-specific candidate using only that timeframe's CSV and the corresponding barrier profile. For example, M1/M5 use `--stop-atr 0.7 --target-atr 1.0`; M15/M30 use `--stop-atr 1.0 --target-atr 1.5`. Use `--candidate-only` to avoid changing the active model registry, and inspect the saved chronological validation metrics and `promotion_rejections` before configuring an artifact for execution. The active API risk profiles use 0.7 ATR stops with 0.8/1.0 ATR scalp targets and 1.0 ATR stops with 1.2/1.5 ATR swing targets, depending on H4 alignment; these are fixed profiles, not learned guarantees. Scalp orders require a non-zero stop loss. The available local CSVs are XAUUSD-only, so they do not establish cross-instrument entry-model quality.

The current local timeframe entry candidates were trained from 2,000 chronologically distributed labels per timeframe, but all failed promotion gates (M1/M5/M15/M30 holdout balanced accuracy: 0.226/0.226/0.181/0.189; gate: 0.55). At the operator's explicit request, the six-timeframe API now loads these candidate artifacts directly despite their rejected validation status. This is an override, not a promotion or evidence of reliability; their weak measured performance means signals should be treated as experimental and trading risk should be controlled accordingly. The separately evaluated pooled trend artifact had 0.815 chronological balanced accuracy across its reported test set, but trend accuracy is not an assurance of profitable entries.

The current H1/M15 reversal candidates were force-promoted despite failing validation gates, and their recorded reversal precision and recall are weak. The staged direction/confirmation flow makes those model outputs explicit and avoids trading without the expected confirmations; it does not improve the trained weights or guarantee a sharp reversal, accurate prediction, or profitable trade. Retrain and promote candidates only after they pass the documented out-of-sample validation gates.

The EA Logs tab records H4 overall direction, H1 current direction, lower-timeframe setup outputs, selected POI, entry-zone status, and the execution gate. Class scores describe model rankings; context cues are separate observable features, not feature attribution or proof of causation. The EA marks the selected structural POI on-chart and checks the current quote against the entry zone before opening a position. Scalp grid spacing uses a trained direction-specific adverse-excursion estimate in ATR units when that regression model is available, bounded between the configured base spacing and three times that spacing; otherwise it explicitly falls back to the configured base interval. The EA title and embedded dashboard version are sourced from the same version constant. Dashboard initialization does not require terminal trading permission, and the EA creates the dashboard before making ML requests. Synchronous ML requests are capped at five seconds and use exponential retry backoff after failures to reduce UI stalls when the service is unavailable. Store the Render `INGEST_API_KEY` in the terminal-local `MQL5/Files/NEXA_ML_API_KEY.txt` file; it is intentionally not stored in source. The `InpLicenseKey` input is checked online at `https://nexafunds-steel.vercel.app/api/licenses/validate` and bound to the current MT5 account number, broker server, and account type. The Vercel route proxies to NexaFunds' existing license-validation API, which verifies the signature and active product/account record. The website license is checked at initialization and before starting trading; use the dashboard recheck button to check it again. In MT5, allow WebRequest access to both `https://nexa-funds-ai.onrender.com` and `https://nexafunds-steel.vercel.app`. An ML API HTTP 401 is shown as `ML AUTH REQUIRED`; an invalid/revoked/account-mismatched website license blocks trading.

## API
`GET /health`
`POST /api/market/candles`
`GET /api/analysis/XAUUSD/M15`
`GET /api/channels/XAUUSD/M15`
`GET /api/supply-demand/XAUUSD/M15`

The analysis endpoints use their configured trend model for the requested timeframe. The EA consensus endpoint retains its H1/M15 compatibility path for older clients; requests containing all six candle arrays use H1 as the mandatory trade direction, treat H4 as optional context, and require M30/M15/M5/M1 entry confluence.

## NexaFunds
Use the existing Node/Express backend as a proxy: React -> NexaFunds API -> Python Analysis AI. The frontend can draw `analysis.primary_channel.upper/lower` and `analysis.supply_demand_zones` directly. The EA should consume the same analysis endpoint/JSON and enforce execution/risk limits separately.

## Learning loop
Every executed setup should be recorded in `learning.db` with its features, model version and eventual R-multiple. Retraining is controlled; do not replace the live model after every trade.
