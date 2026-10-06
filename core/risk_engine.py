from dataclasses import dataclass
@dataclass
class RiskEngine:
    risk_per_trade:float=.01
    def plan(self,signal,price,atr,balance=0):
        if signal=='WAIT' or atr<=0:return {'tradable':False,'stop_loss':None,'take_profit':None}
        d=1.5*atr; reward=3*atr; sl=price-d if signal=='BUY' else price+d; tp=price+reward if signal=='BUY' else price-reward
        return {'tradable':True,'stop_loss':round(sl,8),'take_profit':round(tp,8),'risk_distance':round(d,8),'risk_reward':2.0,'risk_fraction':self.risk_per_trade,'max_loss_currency':round(balance*self.risk_per_trade,2) if balance>0 else None}
