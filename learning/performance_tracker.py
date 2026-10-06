import sqlite3
from pathlib import Path
class PerformanceTracker:
    def __init__(self,path='data/learning.db'): self.path=Path(path)
    def summary(self):
        if not self.path.exists(): return {'trades':0,'win_rate':0,'expectancy_r':0,'profit_factor':0,'max_drawdown_r':0}
        with sqlite3.connect(self.path) as c: rs=[float(x[0]) for x in c.execute('SELECT r_multiple FROM outcomes ORDER BY id').fetchall()]
        if not rs:return {'trades':0,'win_rate':0,'expectancy_r':0,'profit_factor':0,'max_drawdown_r':0}
        eq=peak=dd=0
        for r in rs:eq+=r; peak=max(peak,eq); dd=min(dd,eq-peak)
        gw=sum(r for r in rs if r>0); gl=-sum(r for r in rs if r<0)
        return {'trades':len(rs),'win_rate':sum(r>0 for r in rs)/len(rs),'expectancy_r':sum(rs)/len(rs),'profit_factor':gw/gl if gl else None,'max_drawdown_r':abs(dd)}
