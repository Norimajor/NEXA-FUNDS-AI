"""Train a NexaFunds model from one or many CSV files exported from MT5.
CSV columns: timestamp,open,high,low,close,volume
"""
import argparse
import math
from pathlib import Path

import pandas as pd

from core.data_processor import MarketDataProcessor
from core.feature_engine import FEATURE_NAMES
from ml.dataset_builder import DatasetBuilder
from ml.model_registry import ModelRegistry
from ml.model_trainer import ModelTrainer
from ml.walk_forward import WalkForwardValidator


def load_dataset_from_csvs(paths, max_rows_per_file=50000, sample_stride=1, grid_target_distance=None):
    rows = []
    csv_paths = []
    for path in paths:
        candidate = Path(path)
        if candidate.is_dir():
            csv_paths.extend(sorted(candidate.glob('*.csv')))
        else:
            csv_paths.append(candidate)

    seen = set()
    for csv_path in csv_paths:
        if not csv_path.exists() or csv_path.suffix.lower() != '.csv':
            continue
        if csv_path.resolve() in seen:
            continue
        seen.add(csv_path.resolve())
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f'warning: skipped {csv_path.name}: {exc}')
            continue
        if {'timestamp', 'open', 'high', 'low', 'close', 'volume'}.issubset(df.columns):
            try:
                prepared = MarketDataProcessor().prepare(df)
                builder = DatasetBuilder(
                    horizon=12,
                    stop_atr=1.0,
                    target_atr=1.5,
                    grid_target_distance=grid_target_distance,
                )
                first_index = max(250, builder.horizon * 3)
                eligible_rows = max(0, len(prepared) - first_index - builder.horizon)
                file_stride = max(sample_stride, math.ceil(eligible_rows / max_rows_per_file))
                dataset = builder.build(prepared, sample_stride=file_stride)
                if len(dataset) > max_rows_per_file:
                    dataset = dataset.sample(n=max_rows_per_file, random_state=42).reset_index(drop=True)
                rows.append(dataset)
                print(f'loaded {csv_path.name}: {len(dataset)} rows')
            except Exception as exc:
                print(f'warning: skipped {csv_path.name}: {exc}')

    if not rows:
        raise SystemExit('No usable training rows were produced from the supplied CSV files.')
    combined = pd.concat(rows, ignore_index=True)
    return (
        combined[combined['label'].notna()]
        .sort_values('timestamp', kind='mergesort')
        .reset_index(drop=True)
    )


p = argparse.ArgumentParser()
p.add_argument('paths', nargs='+', help='One or more CSV files or directories to train on.')
p.add_argument('--version', default='model_001')
p.add_argument('--sample-stride', type=int, default=1, help='Use every Nth labeled candle per file (default: 1).')
p.add_argument('--max-rows-per-file', type=int, default=50000, help='Maximum labeled examples per file (default: 50000).')
p.add_argument('--grid-take-profit-distance', type=float, help='Optional price-distance target for direction-specific grid interval labels.')
p.add_argument('--interval-only', action='store_true', help='Train only grid spacing regressors; do not replace or promote a direction classifier.')
args = p.parse_args()

csv_paths = [Path(p) for p in args.paths]
if not csv_paths:
    raise SystemExit('At least one CSV path is required.')

if args.sample_stride < 1:
    raise SystemExit('--sample-stride must be at least 1.')
if args.max_rows_per_file < 200:
    raise SystemExit('--max-rows-per-file must be at least 200.')
if args.grid_take_profit_distance is not None and args.grid_take_profit_distance <= 0:
    raise SystemExit('--grid-take-profit-distance must be greater than zero.')

dataset = load_dataset_from_csvs(
    csv_paths,
    max_rows_per_file=args.max_rows_per_file,
    sample_stride=args.sample_stride,
    grid_target_distance=args.grid_take_profit_distance,
)
if len(dataset) < 200:
    raise SystemExit(f'Only {len(dataset)} labelled examples were produced; supply more historical data.')

print('Label distribution:')
print(dataset['label_name'].value_counts().sort_index().to_string())

X = dataset[FEATURE_NAMES].values
y = dataset.label_name.values
interval_targets = dataset[['grid_interval_buy_atr', 'grid_interval_sell_atr']].values
exit_targets = dataset[[
    'swing_stop_buy_atr', 'swing_target_buy_atr',
    'swing_stop_sell_atr', 'swing_target_sell_atr',
]].values
if args.interval_only:
    result = ModelTrainer().train_intervals(
        X, interval_targets, FEATURE_NAMES, args.version, exit_targets
    )
    print('Interval model:', result)
    raise SystemExit(0)
wf = WalkForwardValidator().validate(X, y)
print('Walk-forward:', wf)
result = ModelTrainer().train(X, y, FEATURE_NAMES, args.version, interval_targets)
print('Training:', result)
ModelRegistry().register(args.version, result['metrics'])
if (
    wf['mean_balanced_accuracy'] >= 0.55
    and result['metrics']['balanced_accuracy'] >= 0.55
    and result['metrics']['roc_auc'] >= 0.55
):
    ModelRegistry().promote(args.version)
    print('Model promoted to models/current.joblib')
else:
    print('Model NOT promoted: out-of-sample quality threshold was not met.')
