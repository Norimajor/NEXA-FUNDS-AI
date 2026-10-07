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

The consensus API reports the planned direction and a structural entry zone, and marks the entry triggered only when the latest M15 candle range intersects that zone. The EA logs the probabilities, model versions, zone, candle range, and trigger state, then gates a new position on that trigger. This is an unconfirmed zone-touch entry and can still catch a falling/rising market; it is not evidence of profitability.

## API
`GET /health`
`POST /api/market/candles`
`GET /api/analysis/XAUUSD/M15`
`GET /api/channels/XAUUSD/M15`
`GET /api/supply-demand/XAUUSD/M15`

## NexaFunds
Use the existing Node/Express backend as a proxy: React -> NexaFunds API -> Python Analysis AI. The frontend can draw `analysis.primary_channel.upper/lower` and `analysis.supply_demand_zones` directly. The EA should consume the same analysis endpoint/JSON and enforce execution/risk limits separately.

## Learning loop
Every executed setup should be recorded in `learning.db` with its features, model version and eventual R-multiple. Retraining is controlled; do not replace the live model after every trade.
