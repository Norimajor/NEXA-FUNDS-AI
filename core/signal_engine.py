class SignalEngine:
    def decide(self,prediction,channel,structure,use_structure_filters=True):
        buy = float(prediction.get('buy_probability', prediction.get('probability_long', 0.0)))
        sell = float(prediction.get('sell_probability', prediction.get('probability_short', 0.0)))
        threshold = float(prediction.get('threshold', .62))
        setup_type = prediction.get('setup_type')
        direction = prediction.get('direction', 'NONE')
        setup_probability = float(prediction.get('setup_probability', max(buy, sell)))
        if setup_type is not None:
            if direction in {'BUY', 'SELL'} and setup_probability >= threshold:
                signal = direction
                signal_reason = 'CONFIDENCE_GATE_PASSED'
            elif direction in {'BUY', 'SELL'}:
                signal = 'WAIT'
                signal_reason = 'LOW_SETUP_CONFIDENCE'
            else:
                signal = 'WAIT'
                signal_reason = 'MODEL_FAVORS_NO_EDGE' if setup_type == 'no_edge' else 'NO_DIRECTIONAL_CLASS'
        else:
            signal = 'WAIT'
            if buy >= threshold and buy > sell:
                signal = 'BUY'
            elif sell >= threshold and sell > buy:
                signal = 'SELL'
            signal_reason = 'CONFIDENCE_GATE_PASSED' if signal != 'WAIT' else 'LOW_DIRECTION_CONFIDENCE'
        if use_structure_filters and channel and ((channel['bias'] == 'BUY' and signal == 'SELL') or (channel['bias'] == 'SELL' and signal == 'BUY') or channel['bias'] == 'RANGE'):
            signal = 'WAIT'
            signal_reason = 'CHANNEL_CONFLICT'
        if use_structure_filters and structure['structure'] == 'BULLISH' and signal == 'SELL':
            signal = 'WAIT'
            signal_reason = 'STRUCTURE_CONFLICT'
        if use_structure_filters and structure['structure'] == 'BEARISH' and signal == 'BUY':
            signal = 'WAIT'
            signal_reason = 'STRUCTURE_CONFLICT'
        payload = {
            'signal': signal,
            'signal_reason': signal_reason,
            'confidence': round(setup_probability * 100, 2),
            'buy_probability': round(buy * 100, 2),
            'sell_probability': round(sell * 100, 2),
            'setup_type': setup_type or ('directional' if signal != 'WAIT' else 'unknown'),
            'direction': direction if direction in {'BUY', 'SELL'} else 'NONE',
            'model_class': prediction.get('model_class', 'unknown'),
            'model_class_probability': float(prediction.get('model_class_probability', setup_probability)),
            'runner_up_class': prediction.get('runner_up_class', 'unknown'),
            'runner_up_probability': float(prediction.get('runner_up_probability', 0.0)),
            'setup_probability': setup_probability,
            'model_version': prediction.get('model_version', 'NO_MODEL'),
            'probability_reversal': float(prediction.get('probability_reversal', 0.0)),
            'probability_continuation': float(prediction.get('probability_continuation', 0.0)),
            'probability_breakout': float(prediction.get('probability_breakout', 0.0)),
            'probability_failed_breakout': float(prediction.get('probability_failed_breakout', 0.0)),
            'probability_no_edge': float(prediction.get('probability_no_edge', 0.0)),
            'probability_long': float(prediction.get('probability_long', buy)),
            'probability_short': float(prediction.get('probability_short', sell)),
            'probability_flat': float(prediction.get('probability_flat', max(0.0, 1.0 - max(buy, sell)))),
            'grid_interval_buy_atr': prediction.get('grid_interval_buy_atr'),
            'grid_interval_sell_atr': prediction.get('grid_interval_sell_atr'),
            'swing_stop_buy_atr': prediction.get('swing_stop_buy_atr'),
            'swing_target_buy_atr': prediction.get('swing_target_buy_atr'),
            'swing_stop_sell_atr': prediction.get('swing_stop_sell_atr'),
            'swing_target_sell_atr': prediction.get('swing_target_sell_atr'),
        }
        for key in (
            'probability_reversal_buy', 'probability_reversal_sell',
            'probability_continuation_buy', 'probability_continuation_sell',
            'probability_breakout_buy', 'probability_breakout_sell',
            'probability_failed_breakout_buy', 'probability_failed_breakout_sell',
        ):
            payload[key] = float(prediction.get(key, 0.0))
        return payload
