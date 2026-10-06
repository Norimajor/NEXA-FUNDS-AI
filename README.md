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

The trainer builds labels from future price movement, trains chronologically, runs walk-forward validation, and only promotes a model when the configured out-of-sample thresholds are met. It never fabricates a signal before a trained model exists.

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
