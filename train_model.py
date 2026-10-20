"""Train a NexaFunds model from one or many CSV files exported from MT5.
CSV columns: timestamp,open,high,low,close,volume
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from core.data_processor import MarketDataProcessor
from core.feature_engine import TRAINING_FEATURE_NAMES, FeatureEngine
from core.market_structure import MarketStructureAnalyzer
from core.trend_engine import MARKET_TREND_LOOKBACK, training_market_trend
from ml.dataset_builder import DatasetBuilder
from ml.execution_levels import ExecutionLevelTrainer
from ml.model_metrics import model_promotion_rejection_reasons
from ml.model_registry import ModelRegistry
from ml.model_trainer import ModelTrainer
from ml.walk_forward import WalkForwardValidator


def load_trend_dataset_from_csvs(paths, max_rows_per_file=1000):
    rows = []
    feature_engine = FeatureEngine()
    structure_analyzer = MarketStructureAnalyzer(lookback=MARKET_TREND_LOOKBACK)
    csv_paths = []
    for path in paths:
        candidate = Path(path)
        csv_paths.extend(sorted(candidate.glob('*.csv')) if candidate.is_dir() else [candidate])

    seen = set()
    for csv_path in csv_paths:
        if not csv_path.exists() or csv_path.suffix.lower() != '.csv':
            continue
        resolved = csv_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        parts = csv_path.stem.split('_m_')
        if len(parts) != 2:
            parts = csv_path.stem.rsplit('_', 1)
        if len(parts) != 2:
            print(
                f'warning: skipped {csv_path.name}: expected SYMBOL_m_TIMEFRAME '
                'or SYMBOL_TIMEFRAME filename'
            )
            continue
        symbol, timeframe = parts[0].upper(), parts[1].upper()
        try:
            prepared = MarketDataProcessor().prepare(pd.read_csv(csv_path))
        except Exception as exc:
            print(f'warning: skipped {csv_path.name}: {exc}')
            continue

        first_index = MARKET_TREND_LOOKBACK - 1
        eligible_count = len(prepared) - first_index
        if eligible_count <= 0:
            print(f'warning: skipped {csv_path.name}: fewer than {MARKET_TREND_LOOKBACK} valid candles')
            continue
        sample_count = min(
            max_rows_per_file,
            max(2, eligible_count // (MARKET_TREND_LOOKBACK // 5)),
        )
        indices = np.unique(
            np.linspace(first_index, len(prepared) - 1, sample_count, dtype=int)
        )
        file_rows = []
        for index in indices:
            history = prepared.iloc[index - MARKET_TREND_LOOKBACK + 1 : index + 1]
            current = history.iloc[-1]
            atr = float(current['atr_14'])
            label, move_atr = training_market_trend(history['close'], atr)
            features = feature_engine.build(
                history,
                None,
                [],
                structure_analyzer.analyze(history),
            )
            if not np.isfinite([features.get(name, 0.0) for name in TRAINING_FEATURE_NAMES]).all():
                continue
            file_rows.append({
                'timestamp': current['timestamp'],
                'instrument': symbol,
                'timeframe': timeframe,
                'trend_label': label,
                'trend_move_atr': move_atr,
                **{name: float(features.get(name, 0.0)) for name in TRAINING_FEATURE_NAMES},
            })
        rows.extend(file_rows)
        print(
            f'loaded {csv_path.name}: {len(file_rows)} trend rows '
            f'from {len(prepared)} valid candles'
        )

    if not rows:
        raise SystemExit('No usable trend-training rows were produced from the supplied CSV files.')
    return (
        pd.DataFrame(rows)
        .sort_values('timestamp', kind='mergesort')
        .reset_index(drop=True)
    )


def load_dataset_from_csvs(
    paths,
    max_rows_per_file=50000,
    sample_stride=1,
    horizon=12,
    stop_atr=1.0,
    target_atr=1.5,
    grid_target_distance=None,
    include_execution_targets=False,
    execution_cost_atr=0.05,
):
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
                    horizon=horizon,
                    stop_atr=stop_atr,
                    target_atr=target_atr,
                    grid_target_distance=grid_target_distance,
                    include_context_detectors=False,
                    include_execution_targets=include_execution_targets,
                    execution_cost_atr=execution_cost_atr,
                )
                first_index = max(MARKET_TREND_LOOKBACK, builder.horizon * 3)
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
p.add_argument('--horizon', type=int, default=12, help='Future candles used to label each setup.')
p.add_argument('--stop-atr', type=float, default=1.0, help='Stop barrier distance in ATR units.')
p.add_argument('--target-atr', type=float, default=1.5, help='Target barrier distance in ATR units.')
p.add_argument('--grid-take-profit-distance', type=float, help='Optional price-distance target for direction-specific grid interval labels.')
p.add_argument('--interval-only', action='store_true', help='Train only grid spacing regressors; do not replace or promote a direction classifier.')
p.add_argument('--execution-levels', action='store_true', help='Train separate safety-gated ML scalp and swing POI/SL/TP models; saves candidates without promotion.')
p.add_argument('--execution-mode', choices=('BOTH', 'SCALP', 'SWING'), default='BOTH', help='Execution profile(s) to train when --execution-levels is set.')
p.add_argument('--execution-cost-atr', type=float, default=0.05, help='Conservative spread/commission/slippage cost assumption in ATR units for execution labels and validation.')
p.add_argument('--trend-only', action='store_true', help='Train a separate BUY/SELL/RANGE trend classifier without changing the setup model.')
p.add_argument('--max-trend-rows-per-file', type=int, default=1000, help='Evenly sampled trend examples per input file (default: 1000).')
p.add_argument('--trend-class-weight', choices=('balanced', 'none'), default='balanced', help='Trend class weighting; use none to preserve natural label frequencies.')
p.add_argument('--trend-range-weight-scale', type=float, default=1.0, help='Scale the balanced RANGE class weight from 0 (exclusive) to 1; lower values emphasize BUY/SELL classification.')
p.add_argument('--candidate-only', action='store_true', help='Save model and metrics without registering or promoting it.')
p.add_argument(
    '--force-promote',
    action='store_true',
    help='Promote even when validation gates fail; metrics and failed gates are saved with the candidate.',
)
args = p.parse_args()

csv_paths = [Path(p) for p in args.paths]
if not csv_paths:
    raise SystemExit('At least one CSV path is required.')

if args.sample_stride < 1:
    raise SystemExit('--sample-stride must be at least 1.')
if args.max_rows_per_file < 200:
    raise SystemExit('--max-rows-per-file must be at least 200.')
if args.horizon < 1:
    raise SystemExit('--horizon must be at least 1.')
if args.stop_atr <= 0 or args.target_atr <= 0:
    raise SystemExit('--stop-atr and --target-atr must be greater than zero.')
if args.grid_take_profit_distance is not None and args.grid_take_profit_distance <= 0:
    raise SystemExit('--grid-take-profit-distance must be greater than zero.')
if args.interval_only and args.trend_only:
    raise SystemExit('--interval-only and --trend-only cannot be used together.')
if args.execution_levels and (args.interval_only or args.trend_only):
    raise SystemExit('--execution-levels cannot be combined with --interval-only or --trend-only.')
if args.execution_cost_atr < 0 or not math.isfinite(args.execution_cost_atr):
    raise SystemExit('--execution-cost-atr must be finite and non-negative.')
if args.execution_levels and (
    len(csv_paths) != 1 or not csv_paths[0].is_file() or csv_paths[0].suffix.lower() != '.csv'
):
    raise SystemExit('--execution-levels requires one timeframe-specific CSV file; train each symbol/timeframe separately.')
if not 0 < args.trend_range_weight_scale <= 1:
    raise SystemExit('--trend-range-weight-scale must be greater than 0 and at most 1.')

if args.trend_only:
    trend_dataset = load_trend_dataset_from_csvs(
        csv_paths, max_rows_per_file=args.max_trend_rows_per_file
    )
    if len(trend_dataset) < 200:
        raise SystemExit(
            f'Only {len(trend_dataset)} trend examples were produced; supply more historical data.'
        )
    label_counts = trend_dataset['trend_label'].value_counts().sort_index()
    print('Trend label distribution:')
    print(label_counts.to_string())
    groups = (
        trend_dataset['instrument'] + '_' + trend_dataset['timeframe']
    ).to_numpy()
    result = ModelTrainer().train_trend(
        trend_dataset[TRAINING_FEATURE_NAMES].to_numpy(dtype=float),
        trend_dataset['trend_label'].to_numpy(),
        TRAINING_FEATURE_NAMES,
        args.version,
        groups=groups,
        class_weight=None if args.trend_class_weight == 'none' else args.trend_class_weight,
        range_weight_scale=args.trend_range_weight_scale,
    )
    result['metrics']['training_configuration'] = {
        'trend_lookback': MARKET_TREND_LOOKBACK,
        'trend_class_weight': args.trend_class_weight,
        'trend_range_weight_scale': args.trend_range_weight_scale,
        'max_trend_rows_per_file': args.max_trend_rows_per_file,
        'sampled_examples': len(trend_dataset),
        'input_files': len({
            f"{symbol}_{timeframe}"
            for symbol, timeframe in zip(
                trend_dataset['instrument'], trend_dataset['timeframe']
            )
        }),
        'instruments': sorted(trend_dataset['instrument'].unique().tolist()),
        'timeframes': sorted(trend_dataset['timeframe'].unique().tolist()),
    }
    Path(result['path']).with_suffix('.json').write_text(
        json.dumps(result['metrics'], indent=2)
    )
    if args.candidate_only:
        result['metrics']['validation_status'] = 'candidate_only'
        Path(result['path']).with_suffix('.json').write_text(
            json.dumps(result['metrics'], indent=2)
        )
    print('Chronological trend-model holdout:', json.dumps(result['metrics'], indent=2))
    print(f'Trend model saved separately: {result["path"]}')
    raise SystemExit(0)

dataset = load_dataset_from_csvs(
    csv_paths,
    max_rows_per_file=args.max_rows_per_file,
    sample_stride=args.sample_stride,
    horizon=args.horizon,
    stop_atr=args.stop_atr,
    target_atr=args.target_atr,
    grid_target_distance=args.grid_take_profit_distance,
    include_execution_targets=args.execution_levels,
    execution_cost_atr=args.execution_cost_atr,
)
if len(dataset) < 200:
    raise SystemExit(f'Only {len(dataset)} labelled examples were produced; supply more historical data.')

if args.execution_levels:
    features = dataset[TRAINING_FEATURE_NAMES].to_numpy(dtype=float)
    future_candles = np.stack(dataset['_future_ohlc_atr'].to_numpy())
    trainer = ExecutionLevelTrainer()
    modes = ('SCALP', 'SWING') if args.execution_mode == 'BOTH' else (args.execution_mode,)
    for mode in modes:
        prefix = mode.lower()
        result = trainer.train(
            features,
            dataset[f'{prefix}_action'].to_numpy(dtype=str),
            dataset[
                [f'{prefix}_{name}' for name in (
                    'entry_offset_atr', 'zone_width_atr', 'stop_atr', 'target_atr'
                )]
            ].to_numpy(dtype=float),
            future_candles,
            TRAINING_FEATURE_NAMES,
            mode,
            f'{args.version}_{prefix}',
            purge=args.horizon,
            cost_atr=args.execution_cost_atr,
        )
        result['metrics']['training_configuration'] = {
            'horizon': args.horizon,
            'sample_stride': args.sample_stride,
            'execution_cost_atr': args.execution_cost_atr,
            'input_file': csv_paths[0].name,
            'objective': 'positive after-cost holdout return subject to stop and drawdown gates',
        }
        Path(result['path']).with_suffix('.json').write_text(
            json.dumps(result['metrics'], indent=2)
        )
        print(f'{mode} execution candidate:', json.dumps(result['metrics'], indent=2))
        print(f'Execution model saved separately: {result["path"]}')
    raise SystemExit(0)

print('Label distribution:')
print(dataset['label_name'].value_counts().sort_index().to_string())

X = dataset[TRAINING_FEATURE_NAMES].values
y = dataset.label_name.values
interval_targets = dataset[['grid_interval_buy_atr', 'grid_interval_sell_atr']].values
exit_targets = dataset[[
    'swing_stop_buy_atr', 'swing_target_buy_atr',
    'swing_stop_sell_atr', 'swing_target_sell_atr',
]].values
if args.interval_only:
    result = ModelTrainer().train_intervals(
        X, interval_targets, TRAINING_FEATURE_NAMES, args.version, exit_targets
    )
    print('Interval model:', result)
    raise SystemExit(0)
wf = WalkForwardValidator().validate(X, y)
print('Walk-forward:', wf)
result = ModelTrainer().train(
    X,
    y,
    TRAINING_FEATURE_NAMES,
    args.version,
    interval_targets,
    swing_targets=exit_targets,
)
print('Training:', result)
result['metrics']['training_configuration'] = {
    'horizon': args.horizon,
    'stop_atr': args.stop_atr,
    'target_atr': args.target_atr,
    'sample_stride': args.sample_stride,
    'max_rows_per_file': args.max_rows_per_file,
}
promotion_rejections = model_promotion_rejection_reasons(wf, result['metrics'])
result['metrics']['promotion_status'] = (
    'forced' if promotion_rejections and args.force_promote
    else 'passed' if not promotion_rejections
    else 'rejected'
)
result['metrics']['forced_promotion'] = bool(promotion_rejections and args.force_promote)
result['metrics']['promotion_rejections'] = promotion_rejections
Path(result['path']).with_suffix('.json').write_text(json.dumps(result['metrics'], indent=2))
if args.candidate_only:
    result['metrics']['validation_status'] = 'passed' if not promotion_rejections else 'rejected'
    result['metrics']['promotion_status'] = 'candidate_only'
    Path(result['path']).with_suffix('.json').write_text(json.dumps(result['metrics'], indent=2))
    print(f'Candidate saved without registry changes: {result["path"]}')
    raise SystemExit(0)
ModelRegistry().register(args.version, result['metrics'])
if not promotion_rejections or args.force_promote:
    if promotion_rejections:
        print('WARNING: forcing promotion despite failed validation: ' + '; '.join(promotion_rejections))
    ModelRegistry().promote(args.version)
    print('Model promoted to models/current.joblib')
else:
    print('Model NOT promoted: ' + '; '.join(promotion_rejections))
