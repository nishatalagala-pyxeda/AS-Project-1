"""Sequential screening of six forecasting targets. Original XLSX is read-only.
Python 3.12; dependencies are the same as ../frontier_experiment/requirements.txt.
Example: python run_targets.py --task all --deps /private/tmp/frontier_experiment_deps
Use --report-only to rebuild summaries without fitting models.
"""
import argparse, sys, os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--source',default='/Users/nishatalagala/Downloads/Frontier HPC & Facility Data.xlsx')
p.add_argument('--deps',default='/private/tmp/frontier_experiment_deps')
p.add_argument('--output',default=str(Path(__file__).resolve().parent))
p.add_argument('--task',default='all',choices=['all','temperature','temperature_events','cooling_transitions','waste_heat','accessory_power','power_ramps'])
p.add_argument('--report-only',action='store_true')
args=p.parse_args();sys.path.insert(0,args.deps)
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/frontier_mpl_cache')
import copy, hashlib, json, platform, re, time, zipfile
import xml.etree.ElementTree as E
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,precision_score,recall_score,f1_score,log_loss
from threadpoolctl import threadpool_limits
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(args.output);OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.set_num_interop_threads(2);threadpool_limits(4)
def log(*a):print(*a,flush=True)
def save_json(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))
TASKS=['temperature','temperature_events','cooling_transitions','waste_heat','accessory_power','power_ramps']
LABELS={'temperature':'Coolant return temperatures','temperature_events':'New high-temperature events',
        'cooling_transitions':'Near-zero-flow transitions','waste_heat':'Waste heat','accessory_power':'Accessory power','power_ramps':'Rapid power changes'}

def load_data():
    source=Path(args.source);sha=hashlib.sha256(source.read_bytes()).hexdigest()
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};rows=[]
    with zipfile.ZipFile(source) as z:
        for _,e in E.iterparse(z.open('xl/worksheets/sheet2.xml'),events=('end',)):
            if e.tag!='{'+ns['s']+'}row':continue
            if int(e.attrib['r'])>2:
                r={}
                for c in e:
                    v=c.find('s:v',ns)
                    if v is not None and c.attrib.get('t') not in ['s','e']:
                        r[re.match('[A-Z]+',c.attrib['r'])[0]]=float(v.text)
                if 'A' in r:rows.append(r)
            e.clear()
    d=pd.DataFrame(rows);t=pd.to_datetime(d.A,unit='D',origin='1899-12-30');rt=t.dt.round('min')
    assert ((t-rt).abs().dt.total_seconds()<=1.01).all()
    d.index=pd.DatetimeIndex(rt);assert d.index.is_unique and d.index.is_monotonic_increasing
    grid=pd.date_range('2023-01-01','2023-12-31 23:50',freq='10min');d=d.reindex(grid)
    cols=['B','C','D','E','F','G','I','O','P','Q','N']
    raw=d[cols].to_numpy(np.float32);fit=raw[grid<pd.Timestamp('2023-08-01')]
    mu=np.nanmean(fit,axis=0);sd=np.nanstd(fit,axis=0);sd[sd==0]=1
    hr=grid.hour.to_numpy()+grid.minute.to_numpy()/60;dow=grid.dayofweek.to_numpy()
    cal=np.column_stack([np.sin(2*np.pi*hr/24),np.cos(2*np.pi*hr/24),np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7)])
    f=np.column_stack([(raw-mu)/sd,cal]).astype(np.float32)
    origins=np.arange(35,len(grid)-6);bad=np.r_[0,np.cumsum(~np.isfinite(raw).all(axis=1))]
    origins=origins[(bad[origins+7]-bad[origins-35])==0]
    X=np.stack([f[i-35:i+1] for i in origins]);origin_t=grid[origins];last_t=grid[origins+6]
    masks={'train':last_t<pd.Timestamp('2023-08-01'),
           'validation':(origin_t>=pd.Timestamp('2023-08-01'))&(last_t<pd.Timestamp('2023-10-01')),
           'test':origin_t>=pd.Timestamp('2023-10-01')}
    assert last_t[masks['train']].max()<origin_t[masks['validation']].min()
    assert last_t[masks['validation']].max()<origin_t[masks['test']].min()
    meta={'source':str(source),'source_sha256':sha,'history_steps':36,'input_columns':cols,
          'calendar_features':['hour_sin','hour_cos','weekday_sin','weekday_cos'],
          'input_notes':'Past calculated waste heat N is included for every model; no future inputs or H/PUE values are used.',
          'splits':{k:{'n':int(m.sum()),'first_origin':str(origin_t[m].min()),'last_origin':str(origin_t[m].max()),'last_target':str(last_t[m].max())} for k,m in masks.items()},
          'normalization':'Training raw rows only; <=1.01 second timestamp offsets normalized; complete 10-minute grid; no imputation; exclude windows with any absent record.',
          'configuration':'Fixed 32-unit LSTM, seed42, Adam .001, MAE or binary cross-entropy; max30 epochs/patience5. Trees: 31 leaves, 200 iterations, .05 learning rate, min30/leaf, L2=1. No tree hyperparameter search.',
          'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'sklearn':sklearn.__version__}}
    save_json(OUT/'manifest.json',meta);np.savez_compressed(OUT/'scaling.npz',mean=mu,std=sd)
    log('DATA',meta['splits'])
    return d,grid,origins,X,masks,meta

class LSTM(nn.Module):
    def __init__(self,inputs,outputs,classification=False,prevalence=None):
        super().__init__();self.lstm=nn.LSTM(inputs,32,batch_first=True);self.output=nn.Linear(32,outputs)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        if classification:
            with torch.no_grad():self.output.bias.fill_(float(np.log(prevalence/(1-prevalence))))
    def forward(self,x):return self.output(self.lstm(x)[0][:,-1,:])

def infer(model,x):
    model.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(x),1024):out.append(model(torch.from_numpy(x[i:i+1024])).numpy())
    return np.concatenate(out)

def train_lstm(x,y,masks,directory,classification=False,seed=42):
    torch.manual_seed(seed);np.random.seed(seed)
    tr,va=masks['train'],masks['validation'];start=time.perf_counter()
    prev=float(y[tr].mean()) if classification else None
    model=LSTM(x.shape[-1],y.shape[-1],classification,prev)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
    loader=DataLoader(TensorDataset(torch.from_numpy(x[tr]),torch.from_numpy(y[tr])),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');state=None;stale=0;history=[]
    for epoch in range(1,31):
        model.train();total=0
        for xb,yb in loader:
            opt.zero_grad();pred=model(xb)
            loss=nn.functional.binary_cross_entropy_with_logits(pred,yb) if classification else nn.functional.l1_loss(pred,yb)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss.detach())*len(xb)
        vp=infer(model,x[va])
        value=float(np.mean(np.logaddexp(0,vp)-y[va]*vp)) if classification else float(np.mean(np.abs(vp-y[va])))
        history.append({'epoch':epoch,'training_loss':total/int(tr.sum()),'validation_loss':value})
        if value<best-1e-5:best=value;state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        log(directory.name,'LSTM epoch',epoch,'validation',round(value,5))
        if stale>=5:break
    model.load_state_dict(state);torch.save({'state_dict':state,'inputs':x.shape[-1],'outputs':y.shape[-1],'classification':classification},directory/'lstm.pt')
    save_json(directory/'training_history.json',history)
    valpred=infer(model,x[va]);testpred=infer(model,x[masks['test']])
    if classification:valpred=1/(1+np.exp(-np.clip(valpred,-40,40)));testpred=1/(1+np.exp(-np.clip(testpred,-40,40)))
    return valpred,testpred,{'seed':seed,'best_epoch':best_epoch,'best_validation_loss':best,'seconds':time.perf_counter()-start}

def paired_ci(errors,times,reference):
    ids=pd.factorize(times.normalize())[0];count=np.bincount(ids);rng=np.random.default_rng(2026)
    draws=rng.integers(0,len(count),(2000,len(count)));out=[]
    for name,err in errors.items():
        sums=np.bincount(ids,weights=err-reference);boot=sums[draws].sum(1)/count[draws].sum(1)
        out.append({'model':name,'mean_loss_difference_vs_baseline':float(np.mean(err-reference)),
                    'day_bootstrap_95pct':[float(v) for v in np.quantile(boot,[.025,.975])]})
    return out

def regression(task,d,grid,o,X,masks):
    directory=OUT/task;directory.mkdir(exist_ok=True);tick=time.perf_counter()
    cols={'temperature':['B','D','F'],'waste_heat':['N'],'accessory_power':['P']}[task]
    labels={'B':'Loop 1','D':'Loop 2','F':'Loop 3','N':'Waste heat','P':'Accessory power'}
    unit='°C' if task=='temperature' else 'MW';h=np.array([1,3,6])
    targets=[];base=[];scales=[];outputs=[]
    for col in cols:
        v=d[col].to_numpy(np.float32);s=float(np.nanstd(v[grid<pd.Timestamp('2023-08-01')]))
        for step in h:
            targets.append(v[o+step]);base.append(v[o]);scales.append(s);outputs.append({'target':labels[col],'horizon_minutes':int(step*10),'unit':unit})
    y=np.column_stack(targets);b=np.column_stack(base);s=np.array(scales,np.float32)
    residual=((y-b)/s).astype(np.float32)
    assert np.isfinite(y).all() and np.isfinite(residual).all()
    vp,lp,training=train_lstm(X,residual,masks,directory)
    te,va,tr=masks['test'],masks['validation'],masks['train'];lstm=lp*s+b[te]
    models=[];tp=[];tree_val=[];tree_tick=time.perf_counter()
    flat_tr=X[tr].reshape(int(tr.sum()),-1);flat_va=X[va].reshape(int(va.sum()),-1);flat_te=X[te].reshape(int(te.sum()),-1)
    for j,item in enumerate(outputs):
        model=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.05,max_iter=200,max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=1,early_stopping=False,random_state=42)
        model.fit(flat_tr,residual[tr,j]);models.append(model);tp.append(model.predict(flat_te));tree_val.append(model.predict(flat_va))
        log(task,'TREE',item['target'],item['horizon_minutes'],'complete')
    trees=np.column_stack(tp)*s+b[te];joblib.dump(models,directory/'trees.joblib')
    preds={'Persistence':b[te],'Trees':trees,'LSTM':lstm};truth=y[te];metrics=[]
    for name,pred in preds.items():
        for j,item in enumerate(outputs):
            e=pred[:,j]-truth[:,j]
            metrics.append(dict(item,model=name,MAE=float(np.abs(e).mean()),RMSE=float(np.sqrt(np.mean(e**2))),bias=float(e.mean()),n=len(e)))
    ci=[]
    for j,item in enumerate(outputs):
        errors={name:np.abs(pred[:,j]-truth[:,j]) for name,pred in preds.items() if name!='Persistence'}
        ci.extend([dict(item,**r) for r in paired_ci(errors,grid[o[te]],np.abs(b[te,j]-truth[:,j]))])
    result={'task':task,'kind':'regression','outputs':outputs,'metrics':metrics,'paired_vs_persistence':ci,
            'lstm_training':training,'tree_seconds':time.perf_counter()-tree_tick,
            'validation_normalized_MAE':{'LSTM':float(np.abs(vp-residual[va]).mean()),'Trees':float(np.abs(np.column_stack(tree_val)-residual[va]).mean())},
            'n_by_split':{k:int(m.sum()) for k,m in masks.items()},'elapsed_seconds':time.perf_counter()-tick}
    save_json(directory/'results.json',result)
    np.savez_compressed(directory/'predictions.npz',truth=truth,persistence=b[te],trees=trees,lstm=lstm,origin_unix_s=grid[o[te]].to_numpy().astype('datetime64[s]').astype('int64'))
    log('FINISHED',task,[(m['target'],m['horizon_minutes'],m['model'],round(m['MAE'],4)) for m in metrics])

def event_data(task,d,grid,o,masks):
    q=d.Q.to_numpy(np.float32);temp=d[['B','D','F']].to_numpy(np.float32);flow=d.J.to_numpy(np.float32)
    train_rows=grid<pd.Timestamp('2023-08-01');eligible=np.ones(len(o),bool)
    if task=='temperature_events':
        thresholds=np.nanquantile(temp[train_rows],.95,axis=0)
        scale=np.nanstd(temp[train_rows],axis=0)
        eligible=np.all(temp[o]<=thresholds,axis=1)
        future=np.stack([temp[o+k] for k in range(1,7)],axis=1)
        y=np.any(future>thresholds,axis=(1,2))
        simple=np.max((temp[o]-thresholds)/scale,axis=1)[:,None]
        details={'definition':'Currently all three return temperatures are at or below their training 95th percentiles; event = any loop exceeds its own threshold in next 60 minutes.',
                 'thresholds_C':thresholds.tolist(),'safety_limit':False,'baseline':'Logistic regression on current maximum standardized distance to threshold; plus constant training prevalence.'}
    elif task=='cooling_transitions':
        threshold=float(.05*np.nanmedian(flow[train_rows]));eligible=flow[o]>threshold
        y=np.any(np.stack([flow[o+k]<=threshold for k in range(1,7)],axis=1),axis=1)
        simple=(flow[o]/threshold)[:,None]
        low=flow<=threshold;episodes=[]
        # Missing records break episodes and are not interpreted as normal flow.
        for phase,begin,end in [('train','2023-01-01','2023-08-01'),('validation','2023-08-01','2023-10-01'),('test','2023-10-01','2024-01-01')]:
            ix=np.flatnonzero(low&(grid>=begin)&(grid<end));groups=np.split(ix,np.flatnonzero(np.diff(ix)>1)+1) if len(ix) else []
            episodes.append({'split':phase,'episodes':[{'start':str(grid[g[0]]),'end':str(grid[g[-1]]),'n':len(g)} for g in groups if len(g)]})
        details={'definition':'Currently normal total flow; event = flow falls below 5% of training median within 60 minutes.',
                 'flow_threshold_gpm':threshold,'safety_limit':False,'episodes':episodes}
    else:
        future=np.stack([np.abs(q[o+k]-q[o]) for k in range(1,4)],axis=1).max(axis=1)
        threshold=float(np.quantile(future[masks['train']],.95));y=future>threshold
        simple=np.column_stack([q[o],np.abs(q[o]-q[o-1]),np.abs(q[o]-q[o-3]),np.std(np.stack([q[o-k] for k in range(6)],axis=1),axis=1)])
        details={'definition':'Within next 30 minutes, at least one total-power reading differs from current power by more than training 95th percentile of maximum 30-minute absolute changes.',
                 'change_threshold_MW':threshold,'baseline':'Logistic regression on current power, last 10/30-minute absolute changes, and past-hour variability; plus constant training prevalence.'}
    return y.astype(np.float32),eligible,simple.astype(np.float32),details

def choose_threshold(y,p):
    thresholds=np.linspace(.005,.995,199)
    f=[f1_score(y,p>=t,zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(f))])

def classification(task,d,grid,o,X,masks):
    directory=OUT/task;directory.mkdir(exist_ok=True);tick=time.perf_counter()
    y,eligible,simple,details=event_data(task,d,grid,o,masks)
    local={k:m&eligible for k,m in masks.items()}
    counts={k:{'n':int(m.sum()),'positives':int(y[m].sum()),'prevalence':float(y[m].mean()) if m.sum() else None} for k,m in local.items()}
    details.update(task=task,kind='classification',counts=counts)
    log('EVENT_AUDIT',task,counts)
    if any(c['positives']==0 or c['positives']==c['n'] for c in counts.values()):
        details['status']='Not fitted: at least one split lacks both positive and negative examples; out-of-time event skill cannot be assessed.'
        save_json(directory/'results.json',details);log('SKIPPED',task,details['status']);return
    tr,va,te=local['train'],local['validation'],local['test'];prev=float(y[tr].mean())
    vp,lp,training=train_lstm(X,y[:,None],local,directory,classification=True)
    tree=HistGradientBoostingClassifier(loss='log_loss',learning_rate=.05,max_iter=200,max_leaf_nodes=31,min_samples_leaf=30,l2_regularization=1,early_stopping=False,random_state=42)
    tree.fit(X[tr].reshape(int(tr.sum()),-1),y[tr])
    tv=tree.predict_proba(X[va].reshape(int(va.sum()),-1))[:,1];tp=tree.predict_proba(X[te].reshape(int(te.sum()),-1))[:,1]
    sm=simple[tr].mean(0);sd=simple[tr].std(0);sd[sd==0]=1
    logistic=LogisticRegression(C=1,max_iter=1000,random_state=42).fit((simple[tr]-sm)/sd,y[tr])
    bv=logistic.predict_proba((simple[va]-sm)/sd)[:,1];bp=logistic.predict_proba((simple[te]-sm)/sd)[:,1]
    preds={'Prevalence':np.full(int(te.sum()),prev),'Simple logistic':bp,'Trees':tp,'LSTM':lp.ravel()}
    vals={'Prevalence':np.full(int(va.sum()),prev),'Simple logistic':bv,'Trees':tv,'LSTM':vp.ravel()}
    metrics=[]
    for name,pred in preds.items():
        threshold=choose_threshold(y[va],vals[name]);dec=pred>=threshold
        metrics.append({'model':name,'AP':float(average_precision_score(y[te],pred)),'ROC_AUC':float(roc_auc_score(y[te],pred)),
                        'Brier':float(brier_score_loss(y[te],pred)),'log_loss':float(log_loss(y[te],pred,labels=[0,1])),
                        'threshold_chosen_on_validation':threshold,'precision':float(precision_score(y[te],dec,zero_division=0)),
                        'recall':float(recall_score(y[te],dec,zero_division=0)),'F1':float(f1_score(y[te],dec,zero_division=0)),
                        'alerts':int(dec.sum()),'n':int(te.sum())})
    # Only the first eligible origin is counted for onset clustering; adjacent positives are correlated.
    onset_times=grid[o[te]][y[te]==1];clusters=0 if not len(onset_times) else int(1+np.sum(np.diff(onset_times.values).astype('timedelta64[s]').astype(int)>600))
    errors={name:(pred-y[te])**2 for name,pred in preds.items() if name not in ['Prevalence','Simple logistic']}
    cis=paired_ci(errors,grid[o[te]],(bp-y[te])**2)
    details.update(status='Fitted',metrics=metrics,lstm_training=training,test_positive_clusters=clusters,
                   paired_Brier_vs_simple_logistic=cis,elapsed_seconds=time.perf_counter()-tick)
    save_json(directory/'results.json',details)
    np.savez_compressed(directory/'predictions.npz',truth=y[te],prevalence=preds['Prevalence'],simple_logistic=bp,trees=tp,lstm=lp.ravel(),origin_unix_s=grid[o[te]].to_numpy().astype('datetime64[s]').astype('int64'))
    joblib.dump({'tree':tree,'logistic':logistic,'simple_mean':sm,'simple_std':sd},directory/'models.joblib')
    log('FINISHED',task,[(m['model'],'AP',round(m['AP'],4),'Brier',round(m['Brier'],4)) for m in metrics])

def report():
    meta=json.loads((OUT/'manifest.json').read_text());results=[]
    for task in TASKS:
        f=OUT/task/'results.json'
        if f.exists():results.append(json.loads(f.read_text()))
    save_json(OUT/'all_results.json',{'manifest':meta,'tasks':results})
    summary=[]
    for r in results:
        task=r['task']
        if task=='temperature':
            summary.append({'Target':LABELS[task],'Finding':'Persistence is best averaged across all loops. Loop 3 LSTM has a small 60-minute gain: 1.090 versus 1.114 °C MAE.'})
        elif task=='temperature_events':
            summary.append({'Target':LABELS[task],'Finding':'Simple logistic has the highest average precision (0.145). At its validation-chosen threshold, precision is 24.5% and recall is 18.1%. Trees and LSTM do not improve average precision; probability errors remain substantial.'})
        elif task=='cooling_transitions':
            summary.append({'Target':LABELS[task],'Finding':'Not evaluable: no qualifying transitions in validation or test. Two near-zero-flow episodes occur in April; only seven eligible positive training windows.'})
        elif task=='waste_heat':
            base=next(m['MAE'] for m in r['metrics'] if m['model']=='Persistence' and m['horizon_minutes']==60)
            initial=next(m['MAE'] for m in r['metrics'] if m['model']=='LSTM' and m['horizon_minutes']==60)
            maes=[initial]+[m['MAE'] for run in r.get('additional_lstm_runs',[]) for m in run['metrics'] if m['horizon_minutes']==60]
            gains=[100*(1-v/base) for v in maes]
            finding=f'LSTM 60-minute MAE: {min(maes):.3f}–{max(maes):.3f} MW versus persistence {base:.3f} MW; {min(gains):.1f}–{max(gains):.1f}% lower across {len(maes)} seed(s). Heat is a calculated quantity, not independently measured.'
            summary.append({'Target':LABELS[task],'Finding':finding})
        elif task=='accessory_power':
            summary.append({'Target':LABELS[task],'Finding':'Strongest continuous-target improvement: trees reduce MAE about 17% at 30 minutes and 13% at 60 minutes. At 60 minutes: 41.0 versus 47.0 kW MAE.'})
        else:
            summary.append({'Target':LABELS[task],'Finding':'Trees have highest average precision (0.183 versus logistic 0.119 and LSTM 0.153). At the validation-chosen threshold, precision is 19.2% and recall is 43.3%; most alert windows are false positives. Probability error is nearly the same as logistic.'})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    colors={'Persistence':'#888888','Trees':'#166b9b','LSTM':'#c56624'}
    for ax,task in zip(axes,['temperature','waste_heat','accessory_power']):
        r=next((x for x in results if x['task']==task),None)
        if r is None:ax.set_visible(False);continue
        frame=pd.DataFrame(r['metrics'])
        for model,color in colors.items():
            m=frame[frame.model==model].groupby('horizon_minutes').MAE.mean()
            ax.plot(m.index,m.values,marker='o',color=color,label=model)
        ax.set(title=LABELS[task],xlabel='Horizon (minutes)',ylabel='MAE (°C)' if task=='temperature' else 'MAE (MW)',xticks=[10,30,60]);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(frameon=False,fontsize=9);fig.suptitle('Held-out forecasts: October–December 2023',fontsize=14)
    fig.savefig(OUT/'regression_comparison.png',dpi=170);plt.close(fig)
    parts=['''<!doctype html><html><head><meta charset="utf-8"><title>Frontier target experiments</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#23303b}table{border-collapse:collapse;font-size:13px}td,th{padding:7px 10px;border-bottom:1px solid #ddd;text-align:right}th{background:#f1f4f6}img{max-width:100%}pre{white-space:pre-wrap;font-size:13px}h1,h2{line-height:1.2}.note{background:#f2f5f7;padding:14px}</style></head><body><h1>Six prediction targets for Frontier</h1>
<p>Sequential screening with fixed model settings. Same chronological split as the earlier power experiment: train January–July, validate August–September, test October–December. Every fitted target compares a 32-unit LSTM (one predeclared seed, 42) with histogram boosted trees. Simple baselines are essential. No test-driven tuning was performed.</p>
<p class="note">Event thresholds are statistical definitions learned from training data. They are not facility operating limits. Forecasts assume input measurements are available without reporting delay.</p><img src="regression_comparison.png" alt="Regression comparison">''']
    parts[0]=parts[0].replace('</style>', '.findings td,.findings th{text-align:left;vertical-align:top}.findings td:first-child{white-space:nowrap}td:first-child,th:first-child{text-align:left}</style>',1)
    parts[0]=parts[0].replace('<img src="regression_comparison.png"', '<h2>Findings</h2>'+pd.DataFrame(summary).to_html(index=False,classes='findings')+'<p>Best next target: accessory power with trees. Waste heat with LSTM shows a smaller gain. Event ranking has some signal, but false positives and probability calibration limit its use.</p><img src="regression_comparison.png"',1)
    parts.append('<p>The temperature panel averages MAE across three loops. Plotted LSTM curves use the original seed42 run; extra waste-heat seeds are reported separately below.</p>')
    for r in results:
        parts.append('<h2>'+LABELS[r['task']]+'</h2>')
        if r['kind']=='regression':
            df=pd.DataFrame(r['metrics']);parts.append(df.round(4).to_html(index=False))
            for h in [10,30,60]:
                m=df[df.horizon_minutes==h].groupby('model').MAE.mean();best=m.idxmin()
                parts.append(f'<p>{h} minutes: lowest observed mean absolute error = {best} ({m[best]:.4f}); '+('averaged across three loops.' if r['task']=='temperature' else 'units are MW.')+'</p>')
            parts.append('<p>Residual forecasting: the target is a change from the current value. Persistence predicts zero change. LSTM outputs are standardized by each target’s training standard deviation; validation uses the corresponding normalized MAE. Trees predict each output separately.</p>')
            if 'additional_lstm_runs' in r:
                parts.append('<h3>Additional LSTM seeds</h3><p>These replications use the same architecture, inputs, validation checkpoint rule, and test set. They were run for waste heat after the initial screening identified a gain. No seed was selected as a replacement for the original result.</p>')
                parts.append(pd.DataFrame([dict(seed=run['seed'],**m) for run in r['additional_lstm_runs'] for m in run['metrics']]).round(4).to_html(index=False))
        else:
            parts.append('<p>'+r['definition']+'</p>')
            parts.append('<pre>'+json.dumps({k:r[k] for k in ['thresholds_C','flow_threshold_gpm','change_threshold_MW','counts','test_positive_clusters','status','episodes'] if k in r},indent=2)+'</pre>')
            if 'metrics' in r:
                parts.append(pd.DataFrame(r['metrics']).round(4).to_html(index=False))
                ap=sorted(r['metrics'],key=lambda m:m['AP'],reverse=True);br=sorted(r['metrics'],key=lambda m:m['Brier'])
                parts.append(f'<p>Highest observed average precision: {ap[0]["model"]}. Lowest observed Brier score: {br[0]["model"]}. AP measures event ranking (higher is better); Brier measures probability error (lower is better). Alert thresholds maximize validation F1 and are then frozen.</p>')
                parts.append('<p>'+r['baseline']+'</p><p>Positive examples can be adjacent windows around the same physical event; their count is not a count of independent events.</p>')
        parts.append('<p><a href="'+r['task']+'/results.json">Detailed results and uncertainty checks</a></p>')
    parts.append('''<h2>Method and limits</h2><p>History = six hours at 10-minute intervals. Inputs: three subloop return temperatures and flows, supply temperature, compute/accessory/total power, past calculated waste heat, hour and weekday sine/cosine terms. Both algorithms receive identical observed histories; tree inputs are flattened sequences. H (average return temperature with division errors) and PUE are excluded.</p>
<p>Normalize timestamp offsets of at most 1.01 seconds; reindex a complete grid; do not impute. Exclude windows missing any of the 36 history or six future grid records. All target windows stay within their split. Historical context before split boundaries is allowed. Training-only normalization, thresholds and baseline fitting. LSTM uses Adam .001, batch256, maximum30 epochs, validation patience5; best checkpoint restored. Trees use 200 iterations, learning rate .05, 31 leaves, minimum30 samples per leaf, L2=1, MAE/log-loss as appropriate.</p>
<p>One initial seed per new target is a screening experiment; additional seeds, where present, are reported separately. Fixed settings were not extensively optimized. Models are not refit on validation data. Day-level bootstrap intervals are exploratory, conditional on this final-quarter test, and do not represent other years or facilities. Failed or unavailable event evaluations are reported explicitly.</p>
<p>Zero-flow records and temperature outliers are retained. Calculated waste heat inherits the workbook’s thermal constants and zero-clipping behavior. A year of data with no workload, weather, sensor calibration or maintenance labels cannot establish causes or safe control settings.</p>''')
    parts.append('<h2>Data provenance</h2><pre>'+json.dumps(meta,indent=2)+'</pre><p>Original workbook unchanged. <a href="run_targets.py">Reproduction code</a> · <a href="all_results.json">All results</a></p></body></html>')
    (OUT/'report.html').write_text(''.join(parts))
    log('REPORT updated',len(results),'tasks')

if __name__=='__main__':
    if args.report_only:report();sys.exit(0)
    d,grid,o,X,masks,meta=load_data()
    for task in TASKS if args.task=='all' else [args.task]:
        log('START',task)
        if task in ['temperature','waste_heat','accessory_power']:regression(task,d,grid,o,X,masks)
        else:classification(task,d,grid,o,X,masks)
        report()
    log('ALL REQUESTED TASKS COMPLETE')
