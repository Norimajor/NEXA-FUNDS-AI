from dataclasses import dataclass
@dataclass
class MarketStructureAnalyzer:
    lookback:int=150
    def analyze(self,df):
        w=df.tail(self.lookback).reset_index(drop=True)
        highs=[float(r.high) for _,r in w.iterrows() if bool(r.swing_high)]
        lows=[float(r.low) for _,r in w.iterrows() if bool(r.swing_low)]
        s='NEUTRAL'
        if len(highs)>=2 and len(lows)>=2:
            hh,hl=highs[-1]>highs[-2],lows[-1]>lows[-2]; lh,ll=highs[-1]<highs[-2],lows[-1]<lows[-2]
            if hh and hl:s='BULLISH'
            elif lh and ll:s='BEARISH'
            elif hh and ll:s='EXPANSION'
            elif lh and hl:s='CONTRACTION'
        return {'structure':s,'last_swing_high':highs[-1] if highs else None,'previous_swing_high':highs[-2] if len(highs)>1 else None,'last_swing_low':lows[-1] if lows else None,'previous_swing_low':lows[-2] if len(lows)>1 else None}
