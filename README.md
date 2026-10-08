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

H1 trend bias is calculated separately from the model's future trade-setup class. Its 20-bar ATR-normalized direction threshold matches the trend-bias definition already used in training labels, with EMA alignment, confirmed swing structure, and regression as supporting evidence. This allows a clear downtrend/uptrend to remain identifiable when the setup classifier says `no_edge`; that output still means there is no ML trade setup yet. The consensus API uses this H1 bias and waits for a directional M15 setup before planning an entry. A counter-bias M15 trade is allowed only when the M15 model classifies it as a reversal or failed breakout; disagreement without one of those learned setup classes remains `WAIT`. A planned trade still waits for its structural entry zone to be touched, and the EA also checks that the live quote remains inside the zone before opening a position. Buy/sell probability totals remain in the API for backwards compatibility but are not used to choose the trade direction or shown as the EA's directional decision.

The current H1/M15 reversal candidates were force-promoted despite failing validation gates, and their recorded reversal precision and recall are weak. The staged direction/confirmation flow makes those model outputs explicit and avoids trading without the expected confirmations; it does not improve the trained weights or guarantee a sharp reversal, accurate prediction, or profitable trade. Retrain and promote candidates only after they pass the documented out-of-sample validation gates.

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
