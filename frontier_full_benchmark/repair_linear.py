"""Recompute completed preliminary Ridge fits in float64; preserve originals.
Safe while non-Ridge fits run, provided selected Ridge directories are complete.
Run once during development and again after confirmation completes.
"""
import sys,json,shutil
from pathlib import Path
sys.argv=['repair_linear','--target','accessory_power']
import run_benchmark as b
from data import load,features,COLS,FOLDS
b.threadpool_limits(1)
root=b.OUT
work=[]
for fold in FOLDS:
    paths=list((root/fold['name']).glob('ridge_*/results.json'))
    if paths and all(json.loads(p.read_text())['training'].get('solver_precision')=='float64' for p in paths):continue
    if len(paths)==4:work.append((fold,paths))
if not work:print('No completed preliminary Ridge fits to repair.');sys.exit(0)
d,raw,grid,o,cal,sha=load(b.args.source)
assert sha==json.loads((root/'manifest.json').read_text())['source_sha256']
index=COLS.index('P');v=d.P.to_numpy(b.np.float32)
truth=v[o[:,None]+b.HORIZONS];base=b.np.repeat(v[o,None],5,axis=1)
configs=b.CONFIGS;b.CONFIGS=[c for c in configs if c['family']=='Ridge'];tabular={}
for cfg in b.CONFIGS:
    key=(cfg['history'],cfg.get('own',False))
    tabular[key]=features(raw,cal,o,cfg['history'],index,own=key[1])
for fold,paths in work:
    for p in paths:
        if json.loads(p.read_text())['training'].get('solver_precision')=='float64':continue
        dest=root/'preliminary_float32_archive'/fold['name']/p.parent.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists(),str(dest)
        shutil.move(str(p.parent),str(dest))
    b.run_fold(fold,d,raw,grid,o,cal,truth,base,index,tabular)
b.CONFIGS=configs
print('Float64 correction complete for',[f['name'] for f,_ in work],flush=True)
if (root/'completion.json').exists():b.report()
