from pathlib import Path
import sqlite3, json
class TradeOutcomeStore:
    def __init__(self,path='data/learning.db'):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.execute('CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY,symbol TEXT,timeframe TEXT,timestamp TEXT,signal TEXT,buy_probability REAL,sell_probability REAL,model_version TEXT,entry REAL,stop_loss REAL,take_profit REAL,features_json TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS outcomes(id INTEGER PRIMARY KEY,prediction_id INTEGER UNIQUE,result TEXT,r_multiple REAL,exit_price REAL,exit_timestamp TEXT)')
    def record_prediction(self,p):
        with sqlite3.connect(self.path) as c:
            cur=c.execute('INSERT INTO predictions(symbol,timeframe,timestamp,signal,buy_probability,sell_probability,model_version,entry,stop_loss,take_profit,features_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)',(p['symbol'],p['timeframe'],p['timestamp'],p['signal'],p['buy_probability'],p['sell_probability'],p['model_version'],p.get('entry'),p.get('stop_loss'),p.get('take_profit'),json.dumps(p.get('features',{}))))
            return cur.lastrowid
    def record_outcome(self,prediction_id,result,r_multiple,exit_price,exit_timestamp):
        with sqlite3.connect(self.path) as c:c.execute('INSERT OR REPLACE INTO outcomes(prediction_id,result,r_multiple,exit_price,exit_timestamp) VALUES(?,?,?,?,?)',(prediction_id,result,r_multiple,exit_price,exit_timestamp))
