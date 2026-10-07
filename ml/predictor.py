from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np


class ModelPredictor:
    def __init__(self, path='models/current.joblib', threshold=.62, interval_path=None):
        self.path = Path(path)
        self.interval_path = Path(interval_path) if interval_path else None
        self.threshold = threshold
        self.bundle = None
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

    @property
    def ready(self):
        return self.bundle is not None

    def reload(self):
        self.bundle = None
        self._load()

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
