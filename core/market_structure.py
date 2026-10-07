from dataclasses import dataclass

import pandas as pd
@dataclass
class MarketStructureAnalyzer:
    lookback:int=150
    def analyze(self,df):
        w=df.tail(self.lookback).reset_index(drop=True)
        high_prices = w["swing_high_price"] if "swing_high_price" in w else w["high"]
        low_prices = w["swing_low_price"] if "swing_low_price" in w else w["low"]
        highs=[float(high_prices.iloc[i]) for i in range(len(w)) if bool(w["swing_high"].iloc[i]) and pd.notna(high_prices.iloc[i])]
        lows=[float(low_prices.iloc[i]) for i in range(len(w)) if bool(w["swing_low"].iloc[i]) and pd.notna(low_prices.iloc[i])]
        s='NEUTRAL'
        if len(highs)>=2 and len(lows)>=2:
            hh,hl=highs[-1]>highs[-2],lows[-1]>lows[-2]; lh,ll=highs[-1]<highs[-2],lows[-1]<lows[-2]
            if hh and hl:s='BULLISH'
            elif lh and ll:s='BEARISH'
            elif hh and ll:s='EXPANSION'
            elif lh and hl:s='CONTRACTION'
        return {'structure':s,'last_swing_high':highs[-1] if highs else None,'previous_swing_high':highs[-2] if len(highs)>1 else None,'last_swing_low':lows[-1] if lows else None,'previous_swing_low':lows[-2] if len(lows)>1 else None}
