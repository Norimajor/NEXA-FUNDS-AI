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

The current H1/M15 reversal candidates were force-promoted despite failing validation gates, and their recorded reversal precision and recall are weak. The staged direction/confirmation flow makes those model outputs explicit and avoids trading without the expected confirmations; it does not improve the trained weights or guarantee a sharp reversal, accurate prediction, or profitable trade. Retrain and promote candidates only after they pass the documented out-of-sample validation gates.

The EA Logs tab records the H1 trend model class/version, M15 top and runner-up setup classes, setup reason, aligned/conflicting market-context cues, selected POI, entry-zone status, and execution gate. Class scores describe model rankings; context cues are separate observable features, not feature attribution or proof of causation. The EA marks the selected structural POI on-chart while retaining its M15 confirmation and live-quote entry checks. Scalp grid spacing uses a trained direction-specific adverse-excursion estimate in ATR units when that regression model is available, bounded between the configured base spacing and three times that spacing; otherwise it explicitly falls back to the configured base interval. The EA title and embedded dashboard version are sourced from the same version constant. Dashboard initialization does not require terminal trading permission, and the EA creates the dashboard before making ML requests. Synchronous ML requests are capped at five seconds and use exponential retry backoff after failures to reduce UI stalls when the service is unavailable. Set the EA input `InpMLApiKey` to the current Render `INGEST_API_KEY`; the key is intentionally not stored in source. An HTTP 401 is shown as `ML AUTH REQUIRED`.

## API
`GET /health`
`POST /api/market/candles`
`GET /api/analysis/XAUUSD/M15`
`GET /api/channels/XAUUSD/M15`
`GET /api/supply-demand/XAUUSD/M15`

The analysis endpoints use the trained H1 or M15 trend model for the requested timeframe. Other timeframes report `RANGE` with an unavailable-model reason rather than inferring direction from indicators.

## NexaFunds
Use the existing Node/Express backend as a proxy: React -> NexaFunds API -> Python Analysis AI. The frontend can draw `analysis.primary_channel.upper/lower` and `analysis.supply_demand_zones` directly. The EA should consume the same analysis endpoint/JSON and enforce execution/risk limits separately.

## Learning loop
Every executed setup should be recorded in `learning.db` with its features, model version and eventual R-multiple. Retraining is controlled; do not replace the live model after every trade.
