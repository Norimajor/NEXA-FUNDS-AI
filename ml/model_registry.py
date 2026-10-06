from pathlib import Path
import json, shutil, datetime
class ModelRegistry:
    def __init__(self,model_dir='models'):
        self.dir=Path(model_dir); self.dir.mkdir(parents=True,exist_ok=True); self.file=self.dir/'registry.json'
        if not self.file.exists(): self.file.write_text(json.dumps({'active':None,'versions':[]},indent=2))
    def read(self): return json.loads(self.file.read_text())
    def register(self,version,metrics):
        d=self.read(); d['versions'].append({'version':version,'metrics':metrics,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}); self.file.write_text(json.dumps(d,indent=2)); return d
    def promote(self,version):
        src=self.dir/f'{version}.joblib'
        if not src.exists(): raise FileNotFoundError(src)
        shutil.copy2(src,self.dir/'current.joblib'); d=self.read(); d['active']=version; self.file.write_text(json.dumps(d,indent=2)); return version
