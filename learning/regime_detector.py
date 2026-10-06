class RegimeDetector:
    def detect(self,df):
        r=df.iloc[-1]; atr=max(float(r.atr_14),1e-12); price=max(float(r.close),1e-12); strength=abs(float(r.ema_20-r.ema_50))/atr
        regime='TRENDING' if strength>1.2 else 'RANGING' if strength<.35 else 'TRANSITION'
        vol=atr/price; vc='HIGH' if vol>.02 else 'LOW' if vol<.005 else 'NORMAL'
        return {'regime':regime,'volatility':vc,'trend_strength':round(strength,3),'atr_percent':round(vol,6)}
