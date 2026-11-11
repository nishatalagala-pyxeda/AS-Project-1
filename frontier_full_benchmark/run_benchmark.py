"""Broad, bounded forecasting benchmark. No source workbook modifications.
Run accessory_power first, then waste_heat. All raw predictions are saved.
"""
import argparse,sys,os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--source',default='/Users/nishatalagala/Downloads/Frontier HPC & Facility Data.xlsx')
p.add_argument('--deps',default='/private/tmp/frontier_experiment_deps')
p.add_argument('--target',choices=['accessory_power','waste_heat'],required=True)
p.add_argument('--output',default=str(Path(__file__).resolve().parent))
p.add_argument('--report-only',action='store_true')
args=p.parse_args();sys.path.insert(0,args.deps)
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('MPLCONFIGDIR','/private/tmp/frontier_mpl_cache')
import copy,json,time,platform
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor,RandomForestRegressor
from sklearn.linear_model import Ridge
import sklearn,joblib
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import load,masks,features,sequence,COLS,HORIZONS,FOLDS
torch.set_num_threads(4);torch.set_num_interop_threads(2);threadpool_limits(4)
ROOT=Path(args.output);OUT=ROOT/args.target;OUT.mkdir(parents=True,exist_ok=True)
def log(*a):print(*a,flush=True)
def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

# Fixed before scoring this benchmark; no large Cartesian hyperparameter search.
CONFIGS=[
 {'id':'ridge_own24h','family':'Ridge','history':144,'own':True},
 {'id':'ridge_1h','family':'Ridge','history':6},
 {'id':'ridge_6h','family':'Ridge','history':36},
 {'id':'ridge_24h','family':'Ridge','history':144},
 {'id':'extra_trees_6h','family':'Extra Trees','history':36},
 {'id':'random_forest_6h','family':'Random forest','history':36},
 {'id':'boosted_1h','family':'Boosted trees','history':6},
 {'id':'boosted_6h','family':'Boosted trees','history':36},
 {'id':'boosted_24h','family':'Boosted trees','history':144},
 {'id':'boosted_own6h','family':'Boosted trees','history':36,'own':True},
 {'id':'lstm_1h','family':'LSTM','history':6},
 {'id':'lstm_6h','family':'LSTM','history':36},
 {'id':'gru_6h','family':'GRU','history':36},
 {'id':'tcn_6h','family':'TCN','history':36},
 {'id':'tcn_24h','family':'TCN','history':144},
]
BASELINES=['Persistence','Trailing median','Daily pattern']

class CausalBlock(nn.Module):
    def __init__(self,inputs,channels,dilation):
        super().__init__();self.pad=2*dilation;self.conv=nn.Conv1d(inputs,channels,3,padding=self.pad,dilation=dilation)
        self.skip=nn.Conv1d(inputs,channels,1) if inputs!=channels else nn.Identity()
        self.dropout=nn.Dropout(.1)
    def forward(self,x):
        z=self.conv(x)[:,:,:-self.pad]
        return torch.relu(self.skip(x)+self.dropout(torch.relu(z)))

class Network(nn.Module):
    def __init__(self,kind,inputs,history):
        super().__init__();self.kind=kind
        if kind=='TCN':
            layers=[];channels=inputs;dilation=1
            while 2*dilation<history*2:
                layers.append(CausalBlock(channels,16,dilation));channels=16;dilation*=2
            self.body=nn.Sequential(*layers);self.head=nn.Linear(16,5)
        else:
            self.body=(nn.LSTM if kind=='LSTM' else nn.GRU)(inputs,32,batch_first=True);self.head=nn.Linear(32,5)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        z=self.body(x.transpose(1,2))[:,:,-1] if self.kind=='TCN' else self.body(x)[0][:,-1,:]
        return self.head(z)

def infer(model,x):
    model.eval();out=[]
    with torch.no_grad():
        for i in range(0,len(x),512):out.append(model(torch.from_numpy(x[i:i+512])).numpy())
    return np.concatenate(out)

def fit_neural(cfg,x,y,mm,directory,seed=42):
    torch.manual_seed(seed);np.random.seed(seed);tick=time.perf_counter()
    previous_threads=torch.get_num_threads()
    if cfg['family']=='TCN':torch.set_num_threads(1)
    model=Network(cfg['family'],x.shape[-1],cfg['history'])
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
    tr,va=mm['train'],mm['validation']
    loader=DataLoader(TensorDataset(torch.from_numpy(x[tr]),torch.from_numpy(y[tr])),batch_size=256,shuffle=True,generator=torch.Generator().manual_seed(seed))
    best=float('inf');stale=0;state=None;history=[]
    for epoch in range(1,21):
        model.train();total=0
        for xb,yb in loader:
            opt.zero_grad();loss=nn.functional.l1_loss(model(xb),yb);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=float(loss.detach())*len(xb)
        value=float(np.abs(infer(model,x[va])-y[va]).mean())
        history.append({'epoch':epoch,'train_normalized_MAE':total/int(tr.sum()),'validation_normalized_MAE':value})
        if value<best-1e-5:best=value;state=copy.deepcopy(model.state_dict());stale=0;best_epoch=epoch
        else:stale+=1
        log(directory.parent.name,cfg['id'],'seed',seed,'epoch',epoch,'validation',round(value,4))
        if stale>=3:break
    model.load_state_dict(state)
    torch.save({'state_dict':state,'config':cfg,'seed':seed},directory/f'model_seed{seed}.pt')
    save(directory/f'learning_seed{seed}.json',history)
    vp=infer(model,x[va]);sp=infer(model,x[mm['score']])
    used_threads=torch.get_num_threads()
    torch.set_num_threads(previous_threads)
    return vp,sp,{'seed':seed,'best_epoch':best_epoch,'validation_normalized_MAE':best,'seconds':time.perf_counter()-tick,'torch_threads':used_threads}

def metrics(truth,pred,baseline,validation_truth,validation_pred,peak):
    # Split-conformal 90% symmetric intervals calibrated only on validation.
    n=len(validation_truth);level=min(1,np.ceil((n+1)*.9)/n)
    radius=np.quantile(np.abs(validation_truth-validation_pred),level,axis=0,method='higher')
    lower=np.maximum(0,pred-radius);upper=pred+radius;out=[]
    for j,h in enumerate(HORIZONS):
        e=pred[:,j]-truth[:,j];ae=np.abs(e);ref=np.abs(baseline[:,j]-truth[:,j]);hi=truth[:,j]>=peak
        out.append({'horizon_minutes':int(h*10),'MAE_MW':float(ae.mean()),'RMSE_MW':float(np.sqrt(np.mean(e**2))),
                    'bias_MW':float(e.mean()),'MAE_relative_to_persistence':float(ae.mean()/ref.mean()),
                    'peak_MAE_MW':float(ae[hi].mean()) if hi.any() else None,'peak_n':int(hi.sum()),
                    'interval90_coverage':float(np.mean((truth[:,j]>=lower[:,j])&(truth[:,j]<=upper[:,j]))),
                    'interval90_mean_width_MW':float((upper[:,j]-lower[:,j]).mean()),'n':len(truth)})
    return out,radius

def run():
    start=time.perf_counter();d,raw,grid,o,cal,sha=load(args.source)
    col='P' if args.target=='accessory_power' else 'N';target_index=COLS.index(col);v=d[col].to_numpy(np.float32)
    truth=v[o[:,None]+HORIZONS];base=np.repeat(v[o,None],5,axis=1)
    manifest={'source':args.source,'source_sha256':sha,'target_column':col,'horizons_minutes':(HORIZONS*10).tolist(),
              'configs':CONFIGS,'folds':FOLDS,'common_origins':len(o),'input_columns':COLS,
              'protocol':'Common origins require 24h+one step of complete history and all next6h readings. No imputation. All five future targets stay within split. Train-only scaling and quantiles; validation-only early stopping, Ridge alpha and interval calibration. No Q4 tuning in this run.',
              'confirmation_caveat':'October–December was inspected in earlier experiments; reused confirmation is exploratory, not a fresh independent lockbox.',
              'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'sklearn':sklearn.__version__}}
    if (OUT/'manifest.json').exists():
        old=json.loads((OUT/'manifest.json').read_text())
        assert old['source_sha256']==sha and old['configs']==CONFIGS and old['horizons_minutes']==manifest['horizons_minutes'],'Existing results use different source/configuration; choose a new output directory.'
    save(OUT/'manifest.json',manifest)
    # Build compact lag/statistic matrices once, independent of labels or splits.
    tabular={}
    for cfg in CONFIGS:
        if cfg['family'] in ['Ridge','Extra Trees','Random forest','Boosted trees']:
            key=(cfg['history'],cfg.get('own',False))
            if key not in tabular:tabular[key]=features(raw,cal,o,*key[:1],target_index,own=key[1])
    fold_counts=[]
    for fold in FOLDS:
        mm=masks(grid,o,fold);counts={k:int(m.sum()) for k,m in mm.items()};log('FOLD',args.target,fold['name'],counts);fold_counts.append(dict(fold=fold['name'],**counts))
    save(OUT/'fold_counts.json',fold_counts)
    # Only development folds are run before champion selection.
    for fold in FOLDS[:-1]:run_fold(fold,d,raw,grid,o,cal,truth,base,target_index,tabular)
    ranking=[];champions={}
    for cfg_id in BASELINES+[c['id'] for c in CONFIGS]:
        fold_results=[json.loads((OUT/fold['name']/cfg_id/'results.json').read_text()) for fold in FOLDS[:-1]]
        for j,h in enumerate(HORIZONS):
            score=float(np.mean([r['metrics'][j]['MAE_relative_to_persistence'] for r in fold_results]))
            ranking.append({'model':cfg_id,'horizon_minutes':int(h*10),'development_mean_relative_MAE':score,
                            'fold_relative_MAE':[r['metrics'][j]['MAE_relative_to_persistence'] for r in fold_results]})
    for h in HORIZONS*10:
        candidates=[r for r in ranking if r['horizon_minutes']==h]
        champions[str(h)]=min(candidates,key=lambda r:r['development_mean_relative_MAE'])
    save(OUT/'development_ranking.json',ranking);save(OUT/'champions_selected_before_confirmation.json',champions)
    log('CHAMPIONS LOCKED',args.target,champions)
    run_fold(FOLDS[-1],d,raw,grid,o,cal,truth,base,target_index,tabular)
    # Replicate only neural configs that win a development horizon, never select seeds on Q4.
    neural_ids={c['id']:c for c in CONFIGS if c['family'] in ['LSTM','GRU','TCN']}
    for cfg_id in sorted({c['model'] for c in champions.values()}&neural_ids.keys()):
        cfg=neural_ids[cfg_id];fold=FOLDS[-1];mm=masks(grid,o,fold);x,mu,sd=sequence(raw,cal,o,cfg['history'],fold['train_end'],grid)
        scale=float(np.nanstd(raw[grid<pd.Timestamp(fold['train_end']),target_index]));y=((truth-base)/scale).astype(np.float32)
        for seed in [43,44]:
            directory=OUT/'confirmation'/cfg_id/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            if (directory/'results.json').exists():continue
            vp,sp,training=fit_neural(cfg,x,y,mm,directory,seed)
            finish(directory,cfg_id,fold,truth,base,mm,np.maximum(0,vp*scale+base[mm['validation']]),np.maximum(0,sp*scale+base[mm['score']]),grid,o,raw,target_index,training)
    save(OUT/'completion.json',{'elapsed_seconds':time.perf_counter()-start,'complete':True})
    report();log('TARGET COMPLETE',args.target)

def run_fold(fold,d,raw,grid,o,cal,truth,base,target_index,tabular):
    mm=masks(grid,o,fold);tr,va,sc=mm['train'],mm['validation'],mm['score'];folder=OUT/fold['name'];folder.mkdir(exist_ok=True)
    v=raw[:,target_index];daily=np.maximum(0,v[o,None]+v[o[:,None]+HORIZONS-144]-v[o-144,None])
    median=np.repeat(np.median(np.stack([v[o-k] for k in range(6)],axis=1),axis=1)[:,None],5,axis=1)
    for name,pred in [('Persistence',base),('Trailing median',median),('Daily pattern',daily)]:
        directory=folder/name;directory.mkdir(exist_ok=True)
        if not (directory/'results.json').exists():finish(directory,name,fold,truth,base,mm,pred[va],pred[sc],grid,o,raw,target_index,{})
    scale=float(np.nanstd(raw[grid<pd.Timestamp(fold['train_end']),target_index]));y=((truth-base)/scale).astype(np.float32)
    seq_cache={}
    for cfg in CONFIGS:
        directory=folder/cfg['id'];directory.mkdir(exist_ok=True)
        if (directory/'results.json').exists():log('RESUME skip',fold['name'],cfg['id']);continue
        tick=time.perf_counter();family=cfg['family'];models=[]
        if family in ['LSTM','GRU','TCN']:
            if cfg['history'] not in seq_cache:seq_cache[cfg['history']]=sequence(raw,cal,o,cfg['history'],fold['train_end'],grid)
            x,mu,sd=seq_cache[cfg['history']]
            vp,sp,training=fit_neural(cfg,x,y,mm,directory)
            np.savez_compressed(directory/'normalization.npz',mean=mu,std=sd,target_scale=scale)
        else:
            x,labels=tabular[(cfg['history'],cfg.get('own',False))]
            if family=='Ridge':
                xx=x.astype(np.float64);mu=xx[tr].mean(0);sd=xx[tr].std(0);sd[sd==0]=1;xx=(xx-mu)/sd
                vals=[];scores=[];alphas=[1.,10.,100.]
                for alpha in alphas:
                    model=Ridge(alpha=alpha).fit(xx[tr],y[tr]);models.append(model);vals.append(model.predict(xx[va]));scores.append(model.predict(xx[sc]))
                errors=np.array([np.abs(p-y[va]).mean(0) for p in vals]);choices=errors.argmin(0)
                vp=np.column_stack([vals[k][:,j] for j,k in enumerate(choices)]);sp=np.column_stack([scores[k][:,j] for j,k in enumerate(choices)])
                training={'alphas_by_horizon':[alphas[k] for k in choices],'solver_precision':'float64'};np.savez_compressed(directory/'normalization.npz',mean=mu,std=sd,target_scale=scale)
            elif family in ['Extra Trees','Random forest']:
                cls=ExtraTreesRegressor if family=='Extra Trees' else RandomForestRegressor
                model=cls(n_estimators=128,max_depth=12,min_samples_leaf=10,max_features=.7,n_jobs=4,random_state=42).fit(x[tr],y[tr])
                models=[model];vp=model.predict(x[va]);sp=model.predict(x[sc]);training={'n_estimators':128,'max_depth':12,'min_samples_leaf':10,'max_features':.7}
            else:
                vals=[];scores=[]
                for j,h in enumerate(HORIZONS):
                    model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=120,max_leaf_nodes=31,min_samples_leaf=30,max_bins=63,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=42).fit(x[tr],y[tr,j])
                    models.append(model);vals.append(model.predict(x[va]));scores.append(model.predict(x[sc]));log(fold['name'],cfg['id'],'horizon',int(h*10),'done')
                vp=np.column_stack(vals);sp=np.column_stack(scores);training={'max_iter':120,'max_bins':63,'max_leaf_nodes':31,'loss':'absolute_error'}
            joblib.dump({'models':models,'config':cfg,'feature_names':labels,'target_scale':scale,'training':training},directory/'models.joblib')
            training['seconds']=time.perf_counter()-tick
        finish(directory,cfg['id'],fold,truth,base,mm,np.maximum(0,vp*scale+base[va]),np.maximum(0,sp*scale+base[sc]),grid,o,raw,target_index,training)

def finish(directory,name,fold,truth,base,mm,vp,sp,grid,o,raw,target_index,training):
    tr,va,sc=mm['train'],mm['validation'],mm['score'];peak=float(np.nanquantile(raw[grid<pd.Timestamp(fold['train_end']),target_index],.95))
    met,radius=metrics(truth[sc],sp,base[sc],truth[va],vp,peak)
    np.savez_compressed(directory/'predictions.npz',truth=truth[sc],predictions=sp,persistence=base[sc],validation_truth=truth[va],validation_predictions=vp,
                        origin_unix_s=grid[o[sc]].to_numpy().astype('datetime64[s]').astype('int64'),interval90_radius_MW=radius)
    save(directory/'results.json',{'model':name,'fold':fold,'metrics':met,'training':training,'peak_threshold_MW':peak,'counts':{k:int(m.sum()) for k,m in mm.items()}})
    log('FINISHED',args.target,fold['name'],name,'MAE',[round(m['MAE_MW'],4) for m in met])

def report():
    manifest=json.loads((OUT/'manifest.json').read_text());ranking=json.loads((OUT/'development_ranking.json').read_text());champions=json.loads((OUT/'champions_selected_before_confirmation.json').read_text())
    all_results=[]
    for fold in FOLDS:
        for model in BASELINES+[c['id'] for c in CONFIGS]:
            f=OUT/fold['name']/model/'results.json'
            if f.exists():all_results.append(json.loads(f.read_text()))
    final=[r for r in all_results if r['fold']['name']=='confirmation'];summary=[];cis=[]
    for j,h in enumerate(HORIZONS*10):
        choice=champions[str(h)]['model'];r=next(r for r in final if r['model']==choice);b=next(r for r in final if r['model']=='Persistence')
        summary.append({'horizon_minutes':int(h),'selected_model':choice,'development_relative_MAE':champions[str(h)]['development_mean_relative_MAE'],**{k:v for k,v in r['metrics'][j].items() if k!='horizon_minutes'}})
        z=np.load(OUT/'confirmation'/choice/'predictions.npz');err=np.abs(z['predictions'][:,j]-z['truth'][:,j])-np.abs(z['persistence'][:,j]-z['truth'][:,j])
        times=pd.to_datetime(z['origin_unix_s'],unit='s');ids=pd.factorize(times.normalize())[0];counts=np.bincount(ids);sums=np.bincount(ids,weights=err)
        draws=np.random.default_rng(2026).integers(0,len(counts),(2000,len(counts)));boot=sums[draws].sum(1)/counts[draws].sum(1)
        cis.append({'horizon_minutes':int(h),'selected_model':choice,'MAE_difference_vs_persistence_MW':float(err.mean()),'day_bootstrap_95pct':[float(v) for v in np.quantile(boot,[.025,.975])]})
    # Neural replication results remain separate; seed42 is never replaced after scoring.
    replicas=[]
    for f in (OUT/'confirmation').glob('*/seed*/results.json'):replicas.append(json.loads(f.read_text()))
    adaptive=json.loads((OUT/'adaptive_intervals.json').read_text()) if (OUT/'adaptive_intervals.json').exists() else None
    save(OUT/'summary.json',{'manifest':manifest,'selected_confirmation':summary,'selected_paired_intervals':cis,'neural_replications':replicas,'development_ranking':ranking,'all_results':all_results,'adaptive_calibration':adaptive})
    frame=pd.DataFrame([dict(model=r['model'],**m) for r in final for m in r['metrics']])
    multiplier=1000 if args.target=='accessory_power' else 1
    unit='kW' if args.target=='accessory_power' else 'MW'
    readable={c['id']:c['family']+' ('+str(c['history']//6)+'h'+(', target only' if c.get('own') else '')+')' for c in CONFIGS}
    readable['Daily pattern']='Daily change (24h-adjusted)'
    display=[]
    for s in summary:
        display.append({'Horizon (min)':s['horizon_minutes'],'Selected model':readable.get(s['selected_model'],s['selected_model']),
                        'MAE ('+unit+')':s['MAE_MW']*multiplier,'Gain vs persistence (%)':100*(1-s['MAE_relative_to_persistence']),
                        'RMSE ('+unit+')':s['RMSE_MW']*multiplier,'90% coverage (%)':100*s['interval90_coverage'],
                        '90% width ('+unit+')':s['interval90_mean_width_MW']*multiplier})
    peak_display=[]
    baseline_result=next(r for r in final if r['model']=='Persistence')
    for j,s in enumerate(summary):
        bp=baseline_result['metrics'][j]['peak_MAE_MW']
        peak_display.append({'Horizon (min)':s['horizon_minutes'],'Selected peak MAE ('+unit+')':None if s['peak_MAE_MW'] is None else s['peak_MAE_MW']*multiplier,
                             'Persistence peak MAE ('+unit+')':None if bp is None else bp*multiplier,'High-demand samples':s['peak_n']})
    ablation=[]
    for own,all_id,family in [('ridge_own24h','ridge_24h','Ridge 24h'),('boosted_own6h','boosted_6h','Boosted trees 6h')]:
        for h in HORIZONS*10:
            a=next(x['development_mean_relative_MAE'] for x in ranking if x['model']==own and x['horizon_minutes']==h)
            b=next(x['development_mean_relative_MAE'] for x in ranking if x['model']==all_id and x['horizon_minutes']==h)
            ablation.append({'Model':family,'Horizon (min)':int(h),'Target-only relative MAE':a,'All-telemetry relative MAE':b,'All-telemetry gain vs target only (%)':100*(1-b/a)})
    chart_models=['Persistence']+list(dict.fromkeys(s['selected_model'] for s in summary if s['selected_model']!='Persistence'))
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans'});fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for model in chart_models:
        ff=frame[frame.model==model]
        axes[0].plot(ff.horizon_minutes,ff.MAE_MW*multiplier,marker='o',label=readable.get(model,model))
        axes[1].plot(ff.horizon_minutes,ff.interval90_coverage,marker='o',label=model)
    if adaptive:
        axes[1].clear()
        axes[1].plot(HORIZONS*10,[s['interval90_coverage'] for s in summary],marker='o',label='Static validation band')
        for days in [7,30]:
            yy=[next(r['coverage'] for r in adaptive['rows'] if r['fold']=='confirmation' and r['window_days']==days and r['horizon_minutes']==int(h)) for h in HORIZONS*10]
            axes[1].plot(HORIZONS*10,yy,marker='o',label=str(days)+'-day rolling band')
        axes[1].legend(fontsize=8,frameon=False)
    for ax in axes:ax.set_xscale('log');ax.set_xticks([10,30,60,120,360],labels=['10','30','60','120','360']);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.set_xlabel('Forecast horizon (minutes)')
    axes[0].set_ylabel('Mean absolute error ('+unit+')');axes[0].legend(fontsize=8,frameon=False)
    axes[1].set_ylabel('Observed coverage of nominal 90% intervals');axes[1].axhline(.9,color='gray',ls='--');axes[1].set_ylim(0,1)
    fig.suptitle(args.target.replace('_',' ').title()+': reused October–December confirmation',fontsize=14);fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
    order=BASELINES+[c['id'] for c in CONFIGS]
    skill=np.array([[next(r['development_mean_relative_MAE'] for r in ranking if r['model']==model and r['horizon_minutes']==int(h)) for h in HORIZONS*10] for model in order])
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
    im=ax.imshow(skill,cmap='RdYlBu_r',vmin=.7,vmax=1.3,aspect='auto')
    ax.set_xticks(range(5),labels=['10','30','60','120','360']);ax.set_yticks(range(len(order)),labels=[readable.get(m,m) for m in order]);ax.set_xlabel('Forecast horizon (minutes)')
    ax.set_title('Development MAE relative to unchanged-value forecast\nBelow 1 is better; colors capped at 0.7 and 1.3')
    for i in range(len(order)):
        for j in range(5):ax.text(j,i,f'{skill[i,j]:.2f}',ha='center',va='center',fontsize=8,color='black')
    fig.colorbar(im,ax=ax,label='Relative MAE');fig.savefig(OUT/'development_comparison.png',dpi=170);plt.close(fig)
    html=['''<!doctype html><html><head><meta charset="utf-8"><title>Forecasting benchmark</title><style>body{font:15px/1.5 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#23303b}table{font-size:12px;border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#f1f4f6}img{max-width:100%}pre{white-space:pre-wrap;font-size:12px}.note{background:#f3f5f7;padding:15px}</style></head><body>''']
    html.append('<h1>'+args.target.replace('_',' ').title()+' forecasting benchmark</h1><p>15 model configurations, three baselines, five horizons and three chronological development backtests, followed by reused October–December confirmation. Candidates were selected separately per horizon using mean relative MAE over development periods.</p>')
    html.append('<p class="note">October–December was already examined in previous experiments. It is not a new independent lockbox. This run locks choices using development results before scoring confirmation. This is a broad bounded benchmark, not an exhaustive search over every algorithm or hyperparameter.</p>')
    html[0]=html[0].replace('</style>','th{max-width:130px;overflow-wrap:anywhere}td{overflow-wrap:anywhere}.scroll{overflow-x:auto}</style>',1)
    html.append('<h2>Development-selected models on confirmation</h2>'+pd.DataFrame(display).round(3).to_html(index=False)+'<img src="comparison.png" alt="Error and interval coverage">')
    html.append('<h2>High-demand performance</h2>'+pd.DataFrame(peak_display).round(3).to_html(index=False)+'<p>High demand is defined by actual target values at or above the training95th percentile. Lower average error across all readings does not establish lower error during peaks. Prediction-band coverage below is measured across all readings, not guaranteed within this high-demand subset.</p>')
    html.append('<h2>Paired daily uncertainty checks</h2>'+pd.DataFrame(cis).to_html(index=False)+'<p>Negative differences favor the selected forecast over persistence. Day-block bootstrap intervals are exploratory and do not capture uncertainty across years or sites. Symmetric prediction intervals are calibrated on validation absolute residuals; time dependence and changing operating conditions can cause coverage below 90%.</p>')
    if adaptive:
        html.append('<h2>Rolling interval calibration</h2><p>'+adaptive['note']+'</p><p>The right-hand chart compares interval coverage for the selected model at each horizon. Rolling windows use an error only when that forecast’s target timestamp has passed. Point forecasts are unchanged. Larger bands can improve coverage at the cost of precision.</p>'+pd.DataFrame([r for r in adaptive['rows'] if r['fold']=='confirmation']).round(4).to_html(index=False)+'<details><summary>Rolling calibration in all development periods</summary>'+pd.DataFrame(adaptive['rows']).round(4).to_html(index=False)+'</details>')
    html.append('<h2>Development ranking</h2><img src="development_comparison.png" alt="Development error ratios by model and horizon"><details><summary>All development scores</summary>'+pd.DataFrame(ranking).round(4).to_html(index=False)+'</details>')
    html.append('<h2>Does additional telemetry help?</h2>'+pd.DataFrame(ablation).round(3).to_html(index=False)+'<p>Positive gain means the additional telemetry improves development MAE relative to a model using only past target values and calendar features. The comparison adds cooling, compute and other power measurements together, so it does not isolate a causal contribution from any one sensor.</p>')
    if args.target=='waste_heat':
        html.append('<p>Target-only waste-heat forecasting still uses the calculated N series, which already incorporates coolant temperatures and flows. This ablation tests whether the raw telemetry adds to that derived history; it does not show that those sensors can be removed.</p>')
    html.append('<details><summary>All confirmation candidates and metrics</summary>'+frame.round(4).to_html(index=False)+'</details>')
    html.append('<details><summary>Neural seed confirmations</summary><pre>'+json.dumps(replicas,indent=2)+'</pre></details>')
    html.append('<h2>Method</h2><p>Every model is evaluated on common origins with complete preceding 24-hour context plus one extra point for daily adjustment, and complete following six hours. This deliberately discards more data around gaps so short/long histories and baselines have identical evaluation samples. All five targets stay within the train, validation or scoring period. No missing-value imputation. Timestamps displaced by at most 1.01 seconds are normalized.</p>')
    html.append('<p>Baselines:Persistence holds the current target value; Trailing median holds the median of the last six readings; Daily pattern adds yesterday’s change over the same horizon to the current level: <code>y[t] + y[t+h-144] - y[t-144]</code>, with h in10-minute steps. The last baseline is an adjusted daily-change forecast, not a plain same-time-yesterday forecast.</p>')
    html.append('<p>All learned models forecast a residual from current target value. Negative forecasts are clipped to zero before scoring. Tabular models use selected past lags and trailing means, standard deviations, extrema and changes; sequence models use the corresponding observed sequence. Own-target ablations use only past target values and calendar features. Models never use future compute power, future temperatures or future flow. Neural networks and forests predict five horizons jointly; boosting fits each horizon separately. Ridge can select a different validation alpha at each horizon.</p>')
    html.append('<p>Ridge alpha is selected independently per horizon from 1,10,100 on validation. Random forest/Extra Trees:128 trees, maximum depth12, minimum10 samples per leaf, 70% features per split. Boosting:120 iterations,31 leaves,63 bins,.05 learning rate,MAE loss. Neural networks:32-unit LSTM/GRU or16-channel causal dilated convolution, up to20 epochs,patience3,Adam .001,MAE loss,seed42. Neural configurations winning any development horizon are repeated with seeds43/44 on confirmation; every seed is retained, with no seed selection.</p>')
    html.append('<p>Report MAE,RMSE,bias,high-demand MAE(training95th-percentile target),relative MAE and nominal90% interval coverage/width. Equal weighting across development periods prevents longer periods dominating model choice. The June–July period includes incomplete source coverage. No causal/control claims. Waste heat remains a workbook-derived estimate with fixed constants and negative-temperature-difference clipping.</p>')
    html.append('<details><summary>Full provenance</summary><pre>'+json.dumps(manifest,indent=2)+'</pre></details><p><a href="summary.json">Machine-readable results</a> · <a href="../run_benchmark.py">Code</a> · <a href="../data.py">Data preparation</a></p></body></html>')
    (OUT/'report.html').write_text(''.join(html));log('REPORT READY',args.target)

if __name__=='__main__':
    if args.report_only:report()
    else:run()
