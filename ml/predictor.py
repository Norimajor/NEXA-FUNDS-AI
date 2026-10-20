from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np

from core.trend_engine import MARKET_TREND_LOOKBACK
from ml.execution_levels import EXECUTION_LEVEL_SCHEMA, MODE_LIMITS


class ModelPredictor:
    def __init__(
        self,
        path='models/current.joblib',
        threshold=.62,
        interval_path=None,
        trend_path=None,
        execution_paths=None,
    ):
        self.path = Path(path)
        self.interval_path = Path(interval_path) if interval_path else None
        self.trend_path = Path(trend_path) if trend_path else None
        self.threshold = threshold
        self.execution_paths = {
            mode: Path(path) for mode, path in (execution_paths or {}).items() if path
        }
        self.execution_bundles = {}
        self.bundle = None
        self.trend_bundle = None
        self._load()

    def _load(self):
        if self.path.exists():
            self.bundle = joblib.load(self.path)
            if self.interval_path and self.interval_path.exists():
                interval_bundle = joblib.load(self.interval_path)
                self.bundle['grid_interval_models'] = interval_bundle.get('grid_interval_models', {})
                self.bundle['swing_exit_models'] = interval_bundle.get('swing_exit_models', {})
                self.bundle['interval_feature_names'] = interval_bundle.get(
                    'feature_names', self.bundle.get('feature_names')
                )
        if self.trend_path and self.trend_path.exists():
            self.trend_bundle = joblib.load(self.trend_path)
        self.execution_bundles = {
            mode: joblib.load(path)
            for mode, path in self.execution_paths.items()
            if path.exists()
        }

    @property
    def ready(self):
        return self.bundle is not None

    def reload(self):
        self.bundle = None
        self.trend_bundle = None
        self.execution_bundles = {}
        self._load()

    def predict_execution_plan(self, features, mode, reference_price, atr):
        unavailable = {
            'available': False,
            'tradable': False,
            'mode': mode,
            'reason': 'VALIDATED_EXECUTION_MODEL_UNAVAILABLE',
        }
        if mode not in MODE_LIMITS:
            raise ValueError(f'Unsupported execution mode: {mode}')
        bundle = self.execution_bundles.get(mode)
        if not bundle:
            return unavailable
        if (
            bundle.get('schema_version') != EXECUTION_LEVEL_SCHEMA
            or bundle.get('mode') != mode
            or bundle.get('validation_status') != 'passed'
        ):
            return {**unavailable, 'reason': 'EXECUTION_MODEL_NOT_VALIDATED'}
        if not isinstance(features, Mapping) or reference_price <= 0 or atr <= 0:
            return {**unavailable, 'reason': 'EXECUTION_FEATURES_OR_PRICE_INVALID'}

        feature_names = bundle.get('feature_names')
        classifier = bundle.get('classifier')
        if not feature_names or classifier is None or not hasattr(classifier, 'predict_proba'):
            raise ValueError('Validated execution model must define a probabilistic action model and feature_names.')
        values = np.asarray(
            [[float(features.get(name, 0.0)) for name in feature_names]], dtype=float
        )
        if not np.isfinite(values).all():
            return {**unavailable, 'reason': 'EXECUTION_FEATURES_NONFINITE'}
        probabilities = np.asarray(classifier.predict_proba(values)[0], dtype=float)
        classes = np.asarray(bundle.get('classes', classifier.classes_), dtype=str)
        if len(classes) != len(probabilities):
            raise ValueError('Execution model classes do not match its probability output.')
        best_index = int(np.argmax(probabilities))
        direction = str(classes[best_index])
        confidence = float(probabilities[best_index])
        if direction not in {'BUY', 'SELL'}:
            return {
                **unavailable,
                'available': True,
                'reason': 'ML_NO_TRADE',
                'confidence': confidence,
                'model_version': bundle.get('version', 'unknown'),
            }
        if confidence < float(bundle.get('minimum_confidence', 1.0)):
            return {
                **unavailable,
                'available': True,
                'reason': 'EXECUTION_CONFIDENCE_BELOW_THRESHOLD',
                'direction': direction,
                'confidence': confidence,
                'model_version': bundle.get('version', 'unknown'),
            }

        side_models = bundle.get('side_models', {}).get(direction)
        if not side_models:
            return {**unavailable, 'reason': 'EXECUTION_LEVEL_REGRESSORS_UNAVAILABLE'}
        predictions = {}
        for name, model in side_models.items():
            predictions[name] = float(model.predict(values)[0])
        offset = float(np.clip(predictions['entry_offset_atr'], -0.75, 0.75))
        offset = min(offset, 0.08) if direction == 'BUY' else max(offset, -0.08)
        width = float(np.clip(predictions['zone_width_atr'], 0.05, 0.5))
        stop_distance = float(np.clip(
            predictions['stop_atr'], 0.2, MODE_LIMITS[mode]['max_stop_atr']
        ))
        target_distance = float(np.clip(predictions['target_atr'], 0.3, 3.0))
        if target_distance < stop_distance * MODE_LIMITS[mode]['min_reward_risk']:
            return {
                **unavailable,
                'available': True,
                'reason': 'ML_REWARD_RISK_BELOW_SAFETY_FLOOR',
                'direction': direction,
                'confidence': confidence,
                'model_version': bundle.get('version', 'unknown'),
            }
        center = reference_price + offset * atr
        half_width = width * atr / 2.0
        stop_anchor = center + half_width if direction == 'BUY' else center - half_width
        stop_loss = (
            stop_anchor - stop_distance * atr
            if direction == 'BUY'
            else stop_anchor + stop_distance * atr
        )
        take_profit = (
            stop_anchor + target_distance * atr
            if direction == 'BUY'
            else stop_anchor - target_distance * atr
        )
        return {
            'available': True,
            'tradable': True,
            'mode': mode,
            'reason': 'VALIDATED_ML_PLAN',
            'direction': direction,
            'confidence': confidence,
            'entry_price': center,
            'entry_zone_low': center - half_width,
            'entry_zone_high': center + half_width,
            'entry_zone_source': f'validated_ml_{mode.lower()}_execution',
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_atr': stop_distance,
            'target_atr': target_distance,
            'model_version': bundle.get('version', 'unknown'),
            'validation_status': bundle.get('validation_status'),
        }

    def predict_trend(self, features):
        if not self.trend_bundle:
            return {
                'direction': 'RANGE',
                'confidence': 0.0,
                'reason': 'TREND_MODEL_UNAVAILABLE',
                'model_version': 'unavailable',
                'runner_up_class': 'unavailable',
                'runner_up_probability': 0.0,
            }
        model_version = self.trend_bundle.get('version', 'unknown')
        if self.trend_bundle.get('trend_lookback') != MARKET_TREND_LOOKBACK:
            return {
                'direction': 'RANGE',
                'confidence': 0.0,
                'reason': (
                    'TREND_MODEL_LOOKBACK_MISMATCH: '
                    f'expected={MARKET_TREND_LOOKBACK}; '
                    f'model={self.trend_bundle.get("trend_lookback", "unknown")}'
                ),
                'model_version': model_version,
                'runner_up_class': 'unavailable',
                'runner_up_probability': 0.0,
            }
        model = self.trend_bundle.get('trend_model')
        feature_names = self.trend_bundle.get('feature_names')
        if model is None or not feature_names or not hasattr(model, 'predict_proba'):
            raise ValueError('Trend model bundle must define a probabilistic model and feature_names.')
        values = np.asarray(
            [[float(features.get(name, 0.0)) for name in feature_names]],
            dtype=float,
        )
        probabilities = np.asarray(model.predict_proba(values)[0], dtype=float)
        classes = self.trend_bundle.get('trend_class_names', getattr(model, 'classes_', None))
        if classes is None or len(classes) != len(probabilities):
            raise ValueError('Trend model classes do not match its probability output.')
        ranked_indices = np.argsort(probabilities)[::-1]
        best_index = int(ranked_indices[0])
        runner_up_index = int(ranked_indices[1]) if len(ranked_indices) > 1 else best_index
        class_directions = {
            'BUY': 'BUY',
            'UPTREND': 'BUY',
            'SELL': 'SELL',
            'DOWNTREND': 'SELL',
            'RANGE': 'RANGE',
        }
        best_class = str(classes[best_index]).strip().upper()
        runner_up_class = str(classes[runner_up_index]).strip().upper()
        direction = class_directions.get(best_class)
        if direction is None:
            raise ValueError(f'Unsupported trend model class: {best_class}')
        confidence = float(probabilities[best_index])
        runner_up_probability = float(probabilities[runner_up_index])
        return {
            'direction': direction,
            'confidence': confidence,
            'reason': (
                f'ML_TREND_{best_class}={confidence:.3f}; '
                f'runner_up_{runner_up_class}={runner_up_probability:.3f}'
            ),
            'model_version': model_version,
            'runner_up_class': runner_up_class,
            'runner_up_probability': runner_up_probability,
        }

    def predict(self, X, features=None):
        empty_result = {
            'buy_probability': 0.0,
            'sell_probability': 0.0,
            'probability_long': 0.0,
            'probability_short': 0.0,
            'probability_flat': 0.0,
            'probability_reversal': 0.0,
            'probability_continuation': 0.0,
            'probability_breakout': 0.0,
            'probability_failed_breakout': 0.0,
            'probability_no_edge': 0.0,
            'setup_type': 'no_model',
            'direction': 'NONE',
            'setup_probability': 0.0,
            'signal': 'WAIT',
            'threshold': self.threshold,
            'model_version': 'NO_MODEL',
            'grid_interval_buy_atr': None,
            'grid_interval_sell_atr': None,
            'swing_stop_buy_atr': None,
            'swing_target_buy_atr': None,
            'swing_stop_sell_atr': None,
            'swing_target_sell_atr': None,
        }
        if not self.bundle:
            return empty_result

        feature_names = self.bundle.get('feature_names')
        if isinstance(features, Mapping) and feature_names:
            arr = np.asarray(
                [[float(features.get(name, 0.0)) for name in feature_names]],
                dtype=float,
            )
        else:
            arr = np.asarray(X, dtype=float).reshape(1, -1)
            if feature_names and arr.shape[1] != len(feature_names):
                raise ValueError(
                    f'Model expects {len(feature_names)} features, received {arr.shape[1]}.'
                )
        interval_feature_names = self.bundle.get('interval_feature_names', feature_names)
        if isinstance(features, Mapping) and interval_feature_names:
            interval_arr = np.asarray(
                [[float(features.get(name, 0.0)) for name in interval_feature_names]],
                dtype=float,
            )
        else:
            interval_arr = arr
        model = self.bundle['model']
        if not hasattr(model, 'predict_proba'):
            return empty_result

        probabilities = np.asarray(model.predict_proba(arr)[0], dtype=float)
        classes = self.bundle.get('class_names')
        if classes is None or len(classes) != len(probabilities):
            classes = getattr(model, 'classes_', None)
        if classes is None or len(classes) != len(probabilities):
            return empty_result

        event_probabilities = {
            'reversal': 0.0,
            'continuation': 0.0,
            'breakout': 0.0,
            'failed_breakout': 0.0,
            'no_edge': 0.0,
        }
        directional_probabilities = {'BUY': 0.0, 'SELL': 0.0}
        class_probabilities = {}
        for class_name, probability in zip(classes, probabilities):
            key = str(class_name).strip().lower().replace('-', '_').replace(' ', '_')
            class_probabilities[key] = float(probability)
            if key == 'no_edge':
                event_probabilities['no_edge'] += float(probability)
                continue

            for event_name in ('reversal', 'continuation', 'breakout', 'failed_breakout'):
                if key == event_name:
                    event_probabilities[event_name] += float(probability)
                elif key in {f'{event_name}_buy', f'{event_name}_sell'}:
                    event_probabilities[event_name] += float(probability)
                    direction = 'BUY' if key.endswith('_buy') else 'SELL'
                    directional_probabilities[direction] += float(probability)

        best_index = int(np.argmax(probabilities))
        best_class = str(classes[best_index]).strip().lower().replace('-', '_').replace(' ', '_')
        setup_probability = float(probabilities[best_index])
        ranked_indices = np.argsort(probabilities)[::-1]
        runner_up_index = int(ranked_indices[1]) if len(ranked_indices) > 1 else best_index
        runner_up_class = (
            str(classes[runner_up_index]).strip().lower().replace('-', '_').replace(' ', '_')
        )
        runner_up_probability = float(probabilities[runner_up_index])
        if best_class.endswith('_buy') or best_class.endswith('_sell'):
            setup_type = best_class.rsplit('_', 1)[0]
            direction = 'BUY' if best_class.endswith('_buy') else 'SELL'
        elif best_class in event_probabilities:
            setup_type = best_class
            direction = 'NONE'
        else:
            setup_type = 'unknown'
            direction = 'NONE'

        signal = direction if direction in {'BUY', 'SELL'} and setup_probability >= self.threshold else 'WAIT'
        buy_probability = directional_probabilities['BUY']
        sell_probability = directional_probabilities['SELL']
        result = {
            'buy_probability': buy_probability,
            'sell_probability': sell_probability,
            'probability_long': buy_probability,
            'probability_short': sell_probability,
            'probability_flat': event_probabilities['no_edge'],
            'probability_reversal': event_probabilities['reversal'],
            'probability_continuation': event_probabilities['continuation'],
            'probability_breakout': event_probabilities['breakout'],
            'probability_failed_breakout': event_probabilities['failed_breakout'],
            'probability_no_edge': event_probabilities['no_edge'],
            'setup_type': setup_type,
            'direction': direction,
            'model_class': best_class,
            'model_class_probability': setup_probability,
            'runner_up_class': runner_up_class,
            'runner_up_probability': runner_up_probability,
            'setup_probability': setup_probability,
            'signal': signal,
            'threshold': self.threshold,
            'model_version': self.bundle.get('version', 'unknown'),
            'grid_interval_buy_atr': None,
            'grid_interval_sell_atr': None,
            'swing_stop_buy_atr': None,
            'swing_target_buy_atr': None,
            'swing_stop_sell_atr': None,
            'swing_target_sell_atr': None,
        }
        interval_models = self.bundle.get('grid_interval_models', {})
        for direction in ('buy', 'sell'):
            interval_model = interval_models.get(direction)
            if interval_model is not None:
                interval = float(interval_model.predict(interval_arr)[0])
                result[f'grid_interval_{direction}_atr'] = float(np.clip(interval, 0.1, 5.0))
        exit_models = self.bundle.get('swing_exit_models', {})
        exit_keys = {
            'buy_stop': 'swing_stop_buy_atr',
            'buy_target': 'swing_target_buy_atr',
            'sell_stop': 'swing_stop_sell_atr',
            'sell_target': 'swing_target_sell_atr',
        }
        for name, result_key in exit_keys.items():
            exit_model = exit_models.get(name)
            if exit_model is not None:
                distance = float(exit_model.predict(interval_arr)[0])
                result[result_key] = float(np.clip(distance, 0.1, 5.0))
        for class_name, probability in class_probabilities.items():
            if class_name.endswith('_buy') or class_name.endswith('_sell'):
                result[f'probability_{class_name}'] = probability
        return result
