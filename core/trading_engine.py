from __future__ import annotations
from dataclasses import dataclass, field
from .channel_detector import ChannelDetector
from .supply_demand_detector import SupplyDemandDetector
from .market_structure import MarketStructureAnalyzer
from .feature_engine import FeatureEngine
from .signal_engine import SignalEngine
from .risk_engine import RiskEngine
from .trend_engine import TrendEngine

@dataclass
class TradingEngine:
    predictor: object
    min_confidence: float = .62
    channel_detector: ChannelDetector = field(default_factory=ChannelDetector)
    sd_detector: SupplyDemandDetector = field(default_factory=SupplyDemandDetector)
    structure_analyzer: MarketStructureAnalyzer = field(default_factory=MarketStructureAnalyzer)
    feature_engine: FeatureEngine = field(default_factory=FeatureEngine)
    signal_engine: SignalEngine = field(default_factory=SignalEngine)
    risk_engine: RiskEngine = field(default_factory=RiskEngine)
    trend_engine: TrendEngine = field(default_factory=TrendEngine)

    def analyze(self, df, account_balance=0, use_structure_filters=True):
        channels=self.channel_detector.detect(df); channel=channels[0] if channels else None
        zones=self.sd_detector.detect(df); structure=self.structure_analyzer.analyze(df)
        trend=self.trend_engine.analyze(df, structure)
        features=self.feature_engine.build(df,channel,zones,structure)
        prediction=self.predictor.predict(self.feature_engine.vector(features),features)
        decision=self.signal_engine.decide(prediction,channel,structure,use_structure_filters)
        price=float(df.iloc[-1].close); atr=float(df.iloc[-1].atr_14)
        risk=self.risk_engine.plan(decision['signal'],price,atr,account_balance)
        recent_closed=df.iloc[:-1].tail(20)
        demand=self.sd_detector.nearest(zones,price,'demand')
        supply=self.sd_detector.nearest(zones,price,'supply')
        swing_low=structure.get('last_swing_low')
        swing_high=structure.get('last_swing_high')
        bullish_level={key:demand[key] for key in ('low','high','mid','distance')} if demand else (
            {'low':float(swing_low),'high':float(swing_low),'mid':float(swing_low),'distance':abs(price-float(swing_low))}
            if swing_low is not None else None
        )
        bearish_level={key:supply[key] for key in ('low','high','mid','distance')} if supply else (
            {'low':float(swing_high),'high':float(swing_high),'mid':float(swing_high),'distance':abs(price-float(swing_high))}
            if swing_high is not None else None
        )
        levels={
            'source':'structural_reference_not_model_price_prediction',
            'potential_bullish_reversal_source':'demand_zone' if demand else ('confirmed_swing_low' if bullish_level else 'unavailable'),
            'potential_bearish_reversal_source':'supply_zone' if supply else ('confirmed_swing_high' if bearish_level else 'unavailable'),
            'potential_bullish_reversal':bullish_level,
            'potential_bearish_reversal':bearish_level,
            'breakout_up':float(recent_closed['high'].max()) if not recent_closed.empty else None,
            'breakout_down':float(recent_closed['low'].min()) if not recent_closed.empty else None,
        }
        probabilities = {'buy': decision['buy_probability'], 'sell': decision['sell_probability']}
        for key in [
            'probability_reversal','probability_continuation','probability_breakout',
            'probability_failed_breakout','probability_no_edge','probability_long',
            'probability_short','probability_flat','probability_reversal_buy',
            'probability_reversal_sell','probability_continuation_buy',
            'probability_continuation_sell','probability_breakout_buy',
            'probability_breakout_sell','probability_failed_breakout_buy',
            'probability_failed_breakout_sell'
        ]:
            if key in decision:
                probabilities[key] = decision[key]
        return {'timestamp':df.iloc[-1].timestamp.isoformat(),'price':price,
                'candle_high':float(df.iloc[-1].high),'candle_low':float(df.iloc[-1].low),'atr':atr,
                'signal':decision['signal'],'signal_reason':decision['signal_reason'],'confidence':decision['confidence'],
            'setup_type':decision['setup_type'],'direction':decision['direction'],
            'model_class':decision['model_class'],
            'model_class_probability':decision['model_class_probability'],
            'runner_up_class':decision['runner_up_class'],
            'runner_up_probability':decision['runner_up_probability'],
            'setup_probability':decision['setup_probability'],
            'trend_direction':trend['direction'],'trend_strength':trend['strength'],
            'trend_reason':trend['reason'],'trend_bull_score':trend['bull_score'],
            'trend_bear_score':trend['bear_score'],
                'probabilities':probabilities,
                'grid_interval_buy_atr':decision['grid_interval_buy_atr'],
                'grid_interval_sell_atr':decision['grid_interval_sell_atr'],
                'swing_stop_buy_atr':decision['swing_stop_buy_atr'],
                'swing_target_buy_atr':decision['swing_target_buy_atr'],
                'swing_stop_sell_atr':decision['swing_stop_sell_atr'],
                'swing_target_sell_atr':decision['swing_target_sell_atr'],
                'model_version':decision['model_version'],'structure':structure,
                'channel_bias':channel['bias'] if channel else 'NONE','primary_channel':channel,
                'channels':channels,'supply_demand_zones':zones,'levels':levels,'risk':risk,'features':features}

# Backwards-compatible public name for existing imports.
TradingAnalysisEngine = TradingEngine
