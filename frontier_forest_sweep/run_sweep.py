"""Bounded forest capacity sweep. Source read only; development-only selection."""
import os,sys
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/frontier_mpl_cache')
sys.path.insert(0,'/private/tmp/frontier_experiment_deps')
from pathlib import Path
CODE_ROOT=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('FRONTIER_FOREST_OUTPUT',str(CODE_ROOT))).resolve()
sys.path.insert(0,str(CODE_ROOT.parent/'frontier_full_benchmark'))
import json,time,hashlib,platform,copy
import numpy as np
import pandas as pd
import joblib,sklearn
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from data import load,masks,features,COLS,HORIZONS,FOLDS
threadpool_limits(4)
SOURCE=os.environ.get('FRONTIER_SOURCE','/Users/nishatalagala/Downloads/Frontier HPC & Facility Data.xlsx')
COUNTS=[1,4,16,64,128,256]
DEPTHS=[2,4,6,8,12]
BUDGETS=[32768,131072,524288]
def save(p,x):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def cfgid(family,own,depth,n):return f'{family}_{"own" if own else "all"}_d{depth}_n{n}'
CONFIGS=[dict(id=cfgid(f,o,d,n),family=f,own=o,depth=d,trees=n) for f in ['RF','ET'] for o in [True,False] for d in DEPTHS for n in COUNTS]
def storage(model):
    leaves=sum(int(np.sum(t.tree_.children_left<0)) for t in model.estimators_)
    nodes=sum(t.tree_.node_count for t in model.estimators_)
    # 16-byte internal nodes, 5 float32 outputs per leaf, 4-byte root per tree.
    return dict(nodes=nodes,leaves=leaves,packed_model_bytes=16*(nodes-leaves)+20*leaves+4*len(model.estimators_))
def measure(truth,pred,base,peak):
    e=pred-truth;mae=np.abs(e).mean(0);ref=np.abs(base-truth).mean(0)
    return dict(mae_MW=mae.tolist(),rmse_MW=np.sqrt((e*e).mean(0)).tolist(),relative_MAE=(mae/ref).tolist(),
                peak_mae_MW=[float(np.abs(e[:,j])[truth[:,j]>=peak].mean()) for j in range(5)],n=len(truth))
def compact(model):
    nodes=[];leaves=[];roots=[]
    def visit(t,k):
        if t.children_left[k]<0:
            idx=len(leaves);leaves.append(t.value[k,:,0].astype(np.float32));return -idx-1
        idx=len(nodes);nodes.append(None)
        l=visit(t,int(t.children_left[k]));r=visit(t,int(t.children_right[k]))
        threshold=np.float32(t.threshold[k])
        # Input features are float32. Round threshold downward to preserve sklearn splits.
        if float(threshold)>t.threshold[k]:threshold=np.nextafter(threshold,np.float32(-np.inf))
        nodes[idx]=(l,r,int(t.feature[k]),threshold);return idx
    for est in model.estimators_:roots.append(visit(est.tree_,0))
    dtype=np.dtype([('left','<i4'),('right','<i4'),('feature','<u4'),('threshold','<f4')])
    return np.array(nodes,dtype=dtype),np.array(leaves,dtype='<f4'),np.array(roots,dtype='<i4')
def compact_predict(nodes,leaves,roots,x):
    out=np.zeros((len(x),5),np.float64)
    for root in roots:
        at=np.full(len(x),root,np.int32)
        while np.any(at>=0):
            rows=np.flatnonzero(at>=0);nd=nodes[at[rows]]
            at[rows]=np.where(x[rows,nd['feature']]<=nd['threshold'],nd['left'],nd['right'])
        out+=leaves[-at-1]
    return out/len(roots)
def export(model,labels,scale,folder,x):
    folder.mkdir(parents=True,exist_ok=True);nodes,leaves,roots=compact(model)
    for name,a in [('nodes',nodes),('leaves',leaves),('roots',roots)]:a.tofile(folder/f'{name}.bin')
    assert sum((folder/f'{s}.bin').stat().st_size for s in ['nodes','leaves','roots'])==storage(model)['packed_model_bytes']
    check=compact_predict(nodes,leaves,roots,x)
    reference=model.predict(x);err=float(np.max(np.abs(check-reference))*scale)
    assert err<1e-4,err
    save(folder/'metadata.json',dict(feature_names=labels,target_scale=scale,horizons_minutes=(HORIZONS*10).tolist(),
        format='little endian; nodes {int32 left,right; uint32 feature; float32 threshold}; leaves float32[5]; roots int32. Negative child = -leaf_index-1.',
        prediction='max(0, current target + target_scale * mean leaf vector)',float32_parity_max_absolute_error_MW=err,
        parity_samples=len(x),**storage(model)))
    joblib.dump(dict(model=model,feature_names=labels,target_scale=scale),folder/'model.joblib')
    # Compile-ready arrays: deployment still needs feature builder and hardware validation.
    lines=['#include <stdint.h>','#include <math.h>','typedef struct { int32_t left,right; uint32_t feature; float threshold; } ForestNode;']
    lines+=['static const ForestNode nodes[] = {']
    lines += ['{%d,%d,%d,%s},'%(v['left'],v['right'],v['feature'],format(float(v['threshold']),'.9g')+'f' if '.' in format(float(v['threshold']),'.9g') or 'e' in format(float(v['threshold']),'.9g') else format(float(v['threshold']),'.9g')+'.0f') for v in nodes]
    lines+=['};','static const float leaves[][5] = {']
    def cf(v):
        s=format(float(v),'.9g');return s+('f' if '.' in s or 'e' in s else '.0f')
    lines+=['{'+','.join(cf(v) for v in row)+'},' for row in leaves]
    lines+=['};','static const int32_t roots[] = {'+','.join(map(str,roots))+'};',
        'void forest_predict(const float *x, float current, float out[5]) {',
        'double sums[5]={0,0,0,0,0};',f'for (unsigned t=0;t<{len(roots)};t++) {{ int32_t k=roots[t];',
        'while(k>=0) { const ForestNode *n=&nodes[k]; k=x[n->feature]<=n->threshold?n->left:n->right; }',
        'for(unsigned h=0;h<5;h++) sums[h]+=leaves[-k-1][h]; }',
        f'for(unsigned h=0;h<5;h++) out[h]=fmaxf(0.0f,current+({cf(scale)})*(float)(sums[h]/{len(roots)}.0));','}']
    (folder/'forest.c').write_text('\n'.join(lines)+'\n')
    # Validate compiled C on the host, including residual scaling and clipping.
    import subprocess,ctypes
    subprocess.run(['cc','-O2','-shared','-fPIC',str(folder/'forest.c'),'-o',str(folder/'forest.so')],check=True,capture_output=True)
    lib=ctypes.CDLL(str(folder/'forest.so'));fn=lib.forest_predict
    arr=np.ctypeslib.ndpointer(dtype=np.float32,ndim=1,flags='C_CONTIGUOUS');fn.argtypes=[arr,ctypes.c_float,arr]
    errors=[];begin=time.perf_counter()
    for row in x[:512]:
        out=np.zeros(5,np.float32);fn(row,ctypes.c_float(1),out)
        errors.append(np.max(np.abs(out-np.maximum(0,1+reference[len(errors)]*scale))))
    elapsed=time.perf_counter()-begin
    assert max(errors)<2e-4,max(errors)
    save(folder/'host_c_verification.json',dict(max_absolute_error_MW=float(max(errors)),samples=len(errors),
        host_microseconds_per_call_including_python_wrapper=elapsed/len(errors)*1e6,
        note='Host validation, not an ESP32 latency, RAM or energy measurement.'))
def run_target(target,d,raw,grid,o,cal,sha):
    folder=ROOT/target;folder.mkdir(exist_ok=True)
    ti=COLS.index('P' if target=='accessory_power' else 'N')
    truth=raw[o[:,None]+HORIZONS,ti].astype(np.float64);base=np.repeat(raw[o,ti,None],5,axis=1).astype(np.float64)
    xx={own:features(raw,cal,o,36,ti,own=own) for own in [True,False]}
    manifest=dict(target=target,source=SOURCE,source_sha256=sha,configs=CONFIGS,folds=FOLDS,horizons_minutes=(HORIZONS*10).tolist(),
        min_samples_leaf=10,max_features=.7,seed=42,history_steps=36,
        selection='Mean relative MAE across three development score periods, equal fold weight. Per-horizon overall and size-budget champions. Size limit applies to maximum packed size across development folds.',
        caveats=['Q4 was inspected previously and is exploratory reused confirmation.','Budgets cover forest arrays only; firmware, feature processing, buffers and runtime are extra.','No ESP32 hardware or power measured.','Bounded capacity sweep; not exhaustive across all forest hyperparameters.'],
        versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__))
    save(folder/'manifest.json',manifest)
    def train_fold(fold,selected=None):
        ff=folder/fold['name'];ff.mkdir(exist_ok=True);mm=masks(grid,o,fold);tr,sc=mm['train'],mm['score']
        scale=float(np.nanstd(raw[grid<pd.Timestamp(fold['train_end']),ti]));y=((truth-base)/scale).astype(np.float32)
        peak=float(np.nanquantile(raw[grid<pd.Timestamp(fold['train_end']),ti],.95))
        for family in ['RF','ET']:
          for own in [True,False]:
            x,labels=xx[own]
            for depth in DEPTHS:
                required=[c for c in CONFIGS if c['family']==family and c['own']==own and c['depth']==depth and (selected is None or c['id'] in selected)]
                if not required:continue
                if all((ff/(c['id']+'.json')).exists() and (selected is None or (ff/c['id']/'host_c_verification.json').exists()) for c in required):continue
                model=(RandomForestRegressor if family=='RF' else ExtraTreesRegressor)(n_estimators=1,max_depth=depth,min_samples_leaf=10,max_features=.7,n_jobs=8,random_state=42,warm_start=True)
                elapsed=0
                for cfg in required:
                    tick=time.perf_counter();model.set_params(n_estimators=cfg['trees']);model.fit(x[tr],y[tr]);elapsed+=time.perf_counter()-tick
                    pred=np.maximum(0,base[sc]+model.predict(x[sc])*scale)
                    result=dict(config=cfg,fold=fold,metrics=measure(truth[sc],pred,base[sc],peak),size=storage(model),fit_seconds_cumulative=elapsed,
                        counts={k:int(v.sum()) for k,v in mm.items()})
                    save(ff/(cfg['id']+'.json'),result)
                    if selected is not None:
                        path=ff/cfg['id'];path.mkdir(exist_ok=True)
                        np.savez_compressed(path/'predictions.npz',truth=truth[sc],predictions=pred,persistence=base[sc],origin_unix_s=grid[o[sc]].to_numpy().astype('datetime64[s]').astype('int64'))
                        export(model,labels,scale,path,x[sc][:1024])
                    print(target,fold['name'],cfg['id'],'MAE',np.round(result['metrics']['mae_MW'],4),'bytes',result['size']['packed_model_bytes'],flush=True)
    for fold in FOLDS[:-1]:train_fold(fold)
    rank=[]
    for cfg in CONFIGS:
        rr=[json.loads((folder/f['name']/(cfg['id']+'.json')).read_text()) for f in FOLDS[:-1]]
        rank.append(dict(config=cfg,development_relative_MAE=np.mean([r['metrics']['relative_MAE'] for r in rr],axis=0).tolist(),
            maximum_development_packed_bytes=max(r['size']['packed_model_bytes'] for r in rr)))
    choices=[]
    for budget in [None]+BUDGETS:
      for j,h in enumerate(HORIZONS*10):
        eligible=[r for r in rank if budget is None or r['maximum_development_packed_bytes']<=budget]
        winner=min(eligible,key=lambda r:r['development_relative_MAE'][j])
        choices.append(dict(budget_bytes=budget,horizon_minutes=int(h),model=winner['config']['id'],development_relative_MAE=winner['development_relative_MAE'][j],maximum_development_packed_bytes=winner['maximum_development_packed_bytes']))
    save(folder/'development_ranking.json',rank);save(folder/'selection_before_confirmation.json',choices)
    joint=[]
    for budget in [None]+BUDGETS:
        eligible=[r for r in rank if budget is None or r['maximum_development_packed_bytes']<=budget]
        winner=min(eligible,key=lambda r:np.mean(r['development_relative_MAE']))
        joint.append(dict(budget_bytes=budget,model=winner['config']['id'],development_mean_relative_MAE=float(np.mean(winner['development_relative_MAE'])),packed_bytes=winner['maximum_development_packed_bytes']))
    save(folder/'joint_selection_before_confirmation.json',joint)
    selected={c['model'] for c in choices+joint};print('SELECTION LOCKED',target,len(selected),'models',flush=True)
    train_fold(FOLDS[-1],selected)
    summary=[]
    for choice in choices:
        r=json.loads((folder/'confirmation'/(choice['model']+'.json')).read_text());j=list(HORIZONS*10).index(choice['horizon_minutes'])
        summary.append(dict(**choice,confirmation_MAE_MW=r['metrics']['mae_MW'][j],confirmation_relative_MAE=r['metrics']['relative_MAE'][j],
            confirmation_packed_bytes=r['size']['packed_model_bytes'],confirmation_budget_met=choice['budget_bytes'] is None or r['size']['packed_model_bytes']<=choice['budget_bytes']))
    save(folder/'summary.json',summary)
    for item in joint:
        r=json.loads((folder/'confirmation'/(item['model']+'.json')).read_text())
        item.update(confirmation_mae_MW=r['metrics']['mae_MW'],confirmation_relative_MAE=r['metrics']['relative_MAE'],confirmation_packed_bytes=r['size']['packed_model_bytes'])
    save(folder/'joint_summary.json',joint)
    print('TARGET COMPLETE',target,flush=True)
if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    if ROOT!=CODE_ROOT:
        import shutil
        for name in ['run_sweep.py','build_report.py']:shutil.copyfile(CODE_ROOT/name,ROOT/name)
    tick=time.perf_counter();d,raw,grid,o,cal,sha=load(SOURCE)
    for target in ['accessory_power','waste_heat']:run_target(target,d,raw,grid,o,cal,sha)
    assert hashlib.sha256(Path(SOURCE).read_bytes()).hexdigest()==sha
    save(ROOT/'completion.json',dict(complete=True,seconds=time.perf_counter()-tick,source_sha256=sha))
