"""Reproducible, chronological LSTM-versus-trees forecasting experiment.

Run with the bundled Python and --deps /private/tmp/frontier_experiment_deps.
Dependencies: numpy, pandas, torch, scikit-learn, matplotlib, threadpoolctl.
Reads the original XLSX without modifying it. No pickle input is trusted.
"""
import argparse, sys, os
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--source',default='/Users/nishatalagala/Downloads/Frontier HPC & Facility Data.xlsx')
ap.add_argument('--deps',default='/private/tmp/frontier_experiment_deps')
ap.add_argument('--output',default=str(Path(__file__).resolve().parent))
args=ap.parse_args()
sys.path.insert(0,args.deps)
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/frontier_mpl_cache')
os.environ.setdefault('OMP_NUM_THREADS','4')
import json, time, hashlib, copy, platform, zipfile, re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(args.output); OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
threadpool_limits(limits=4)
START=time.perf_counter()
def log(*s): print(*s,flush=True)
source=Path(args.source)
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
records=[]
with zipfile.ZipFile(source) as z:
    for _,e in ET.iterparse(z.open('xl/worksheets/sheet2.xml'),events=('end',)):
        if e.tag!='{'+ns['s']+'}row': continue
        if int(e.attrib['r'])>2:
            row={}
            for c in e:
                col=re.match('[A-Z]+',c.attrib['r'])[0]
                v=c.find('s:v',ns)
                if v is not None and c.attrib.get('t') not in ['s','e']:
                    row[col]=float(v.text)
            if 'A' in row: records.append(row)
        e.clear()
d=pd.DataFrame(records)
times=pd.to_datetime(d.A,unit='D',origin='1899-12-30')
rounded=times.dt.round('min')
assert ((times-rounded).abs().dt.total_seconds()<=1.01).all()
d.index=pd.DatetimeIndex(rounded)
assert d.index.is_unique and d.index.is_monotonic_increasing
grid=pd.date_range('2023-01-01','2023-12-31 23:50',freq='10min')
d=d.reindex(grid)
cols=['B','C','D','E','F','G','I','O','P','Q']
names=['loop1_return_C','loop1_flow_gpm','loop2_return_C','loop2_flow_gpm',
       'loop3_return_C','loop3_flow_gpm','supply_C','compute_MW','accessory_MW','total_MW']
raw=d[cols].to_numpy(dtype=np.float32)
hour=grid.hour.to_numpy()+grid.minute.to_numpy()/60
dow=grid.dayofweek.to_numpy()
calendar=np.column_stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                          np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7)]).astype(np.float32)
train_end=pd.Timestamp('2023-08-01')
val_end=pd.Timestamp('2023-10-01')
fit=raw[grid<train_end]
mu=np.nanmean(fit,axis=0); sd=np.nanstd(fit,axis=0); sd[sd==0]=1
features=np.column_stack([(raw-mu)/sd,calendar]).astype(np.float32)
lookback=36; horizons=np.array([1,3,6]); q=raw[:,-1]
origins=np.arange(lookback-1,len(grid)-horizons.max())
# All 36 history steps plus six future grid positions must be present.
present=np.isfinite(raw).all(axis=1)
bad=np.r_[0,np.cumsum(~present)]
valid=(bad[origins+7]-bad[origins-35])==0
origins=origins[valid]
X=np.stack([features[i-35:i+1] for i in origins])
Y=q[origins[:,None]+horizons]
base=q[origins]
target_scale=float(sd[-1])
delta=((Y-base[:,None])/target_scale).astype(np.float32)
origin_times=grid[origins]
last_times=grid[origins+6]
masks={'train':last_times<train_end,
       'validation':(origin_times>=train_end)&(last_times<val_end),
       'test':origin_times>=val_end}
assert not np.any(masks['train']&masks['validation'])
assert last_times[masks['train']].max()<origin_times[masks['validation']].min()
assert last_times[masks['validation']].max()<origin_times[masks['test']].min()
data={k:(X[m],delta[m],Y[m],base[m],origin_times[m]) for k,m in masks.items()}
log('SAMPLES',{k:len(v[0]) for k,v in data.items()},'inputs',X.shape)
log('SPLITS',{k:[str(v[4].min()),str(v[4].max())] for k,v in data.items()})
np.savez_compressed(OUT/'scaling.npz',mean=mu,std=sd)

def predict_lstm(model,x):
    model.eval(); parts=[]
    with torch.no_grad():
        for start in range(0,len(x),1024):
            parts.append(model(torch.from_numpy(x[start:start+1024])).numpy())
    return np.concatenate(parts)

class ForecastLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(X.shape[2],32,batch_first=True)
        self.output=nn.Linear(32,3)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    def forward(self,x):
        out,_=self.lstm(x)
        return self.output(out[:,-1,:])

history=[]; lstm_preds=[]; lstm_val=[]; lstm_times=[]
for seed in [42,43,44]:
    torch.manual_seed(seed); np.random.seed(seed)
    model=ForecastLSTM()
    optim=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(torch.from_numpy(data['train'][0]),torch.from_numpy(data['train'][1])),
                      batch_size=256,shuffle=True,generator=generator)
    best=float('inf'); best_state=None; stale=0; tick=time.perf_counter()
    for epoch in range(1,31):
        model.train(); total=0
        for xb,yb in loader:
            optim.zero_grad(); pred=model(xb)
            loss=nn.functional.l1_loss(pred,yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1)
            optim.step(); total+=float(loss.detach())*len(xb)
        vp=predict_lstm(model,data['validation'][0])
        val=float(np.mean(np.abs(vp-data['validation'][1]))*target_scale)
        history.append({'seed':seed,'epoch':epoch,'train_MAE_MW':total/len(data['train'][0])*target_scale,'validation_MAE_MW':val})
        if val<best-1e-5:
            best=val; best_state=copy.deepcopy(model.state_dict()); stale=0; best_epoch=epoch
        else: stale+=1
        log('LSTM',seed,'epoch',epoch,'validation_MAE_MW',round(val,4))
        if stale>=5: break
    model.load_state_dict(best_state)
    elapsed=time.perf_counter()-tick; lstm_times.append(elapsed)
    torch.save({'state_dict':best_state,'input_size':X.shape[2],'hidden_size':32,'feature_names':names+['hour_sin','hour_cos','weekday_sin','weekday_cos']},OUT/f'lstm_seed_{seed}.pt')
    lstm_preds.append(predict_lstm(model,data['test'][0])*target_scale+data['test'][3][:,None])
    lstm_val.append({'seed':seed,'validation_MAE_MW':best,'best_epoch':best_epoch,'seconds':elapsed})
    (OUT/'training_history.json').write_text(json.dumps(history,indent=2))

tree_candidates=[]; best_tree=None; best_score=float('inf')
for leaves in [15,31]:
    tick=time.perf_counter(); models=[]; val_preds=[]
    for j,h in enumerate(horizons):
        model=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.05,
                    max_iter=200,max_leaf_nodes=leaves,min_samples_leaf=30,l2_regularization=1,
                    early_stopping=False,random_state=42)
        model.fit(data['train'][0].reshape(len(data['train'][0]),-1),data['train'][1][:,j])
        models.append(model)
        val_preds.append(model.predict(data['validation'][0].reshape(len(data['validation'][0]),-1)))
        log('TREE leaves',leaves,'horizon',int(h)*10,'finished')
    vp=np.column_stack(val_preds)
    score=float(np.mean(np.abs(vp-data['validation'][1]))*target_scale)
    tree_candidates.append({'max_leaf_nodes':leaves,'validation_MAE_MW':score,'seconds':time.perf_counter()-tick})
    if score<best_score: best_score=score; best_tree=models; chosen_leaves=leaves
    log('TREE candidate',leaves,'validation_MAE_MW',round(score,4))

xt=data['test'][0].reshape(len(data['test'][0]),-1)
tree_pred=np.column_stack([m.predict(xt) for m in best_tree])*target_scale+data['test'][3][:,None]
import joblib
joblib.dump(best_tree,OUT/'trees.joblib')
truth=data['test'][2]; baseline=np.repeat(data['test'][3][:,None],3,axis=1)
predictions={'Persistence':baseline,'Boosted trees':tree_pred}
for seed,p in zip([42,43,44],lstm_preds): predictions[f'LSTM seed {seed}']=p
predictions['LSTM ensemble']=np.mean(lstm_preds,axis=0)
peak_threshold=float(np.nanquantile(fit[:,-1],.95))
metrics=[]
for name,p in predictions.items():
    for j,h in enumerate(horizons):
        err=p[:,j]-truth[:,j]; peak=truth[:,j]>=peak_threshold
        metrics.append({'model':name,'horizon_minutes':int(h)*10,'MAE_MW':float(np.abs(err).mean()),
                        'RMSE_MW':float(np.sqrt(np.mean(err**2))),'bias_MW':float(err.mean()),
                        'peak_MAE_MW':float(np.abs(err[peak]).mean()),'peak_n':int(peak.sum()),'n':len(err)})
# Paired calendar-day bootstrap; keeps within-day correlated errors together.
days=data['test'][4].normalize(); day_ids=pd.factorize(days)[0]; rng=np.random.default_rng(2026)
intervals=[]
for j,h in enumerate(horizons):
    seed_mae=[float(np.mean(np.abs(p[:,j]-truth[:,j]))) for p in lstm_preds]
    for name,p in [('LSTM seed 42',lstm_preds[0]),('LSTM ensemble',predictions['LSTM ensemble'])]:
        diff=np.abs(p[:,j]-truth[:,j])-np.abs(tree_pred[:,j]-truth[:,j])
        sums=np.bincount(day_ids,weights=diff); counts=np.bincount(day_ids)
        draws=rng.integers(0,len(sums),(2000,len(sums)))
        boot=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
        intervals.append({'model':name,'horizon_minutes':int(h)*10,'LSTM_minus_tree_MAE_MW':float(diff.mean()),
                         'day_bootstrap_95pct':[float(v) for v in np.quantile(boot,[.025,.975])],
                         'individual_seed_MAE_range_MW':[min(seed_mae),max(seed_mae)]})
result={'source':str(source),'source_sha256':source_sha,'target':'Frontier total power (MW)',
        'lookback_steps':36,'horizons_minutes':[10,30,60],'features':names,
        'calendar_features':['hour_sin','hour_cos','weekday_sin','weekday_cos'],
        'timestamp_handling':'Round <=1.01 second offsets to nearest minute; reindex 10-minute grid; no imputation; exclude any window with missing input or future-grid record.',
        'splits':{k:{'n':len(v[0]),'origin_start':str(v[4].min()),'origin_end':str(v[4].max()),'last_target':str(grid[origins[masks[k]]+6].max())} for k,v in data.items()},
        'lstm':{'architecture':'LSTM(14 inputs,32 hidden) -> Linear(32,3); standardized residual targets; MAE loss',
                'max_epochs':30,'patience':5,'batch_size':256,'learning_rate':.001,'runs':lstm_val},
        'trees':{'algorithm':'HistGradientBoostingRegressor','loss':'absolute_error','max_iter':200,'learning_rate':.05,
                 'input':'Same 36 x 14 input values, flattened','chosen_max_leaf_nodes':chosen_leaves,'candidates':tree_candidates},
        'peak_threshold_MW_training_95pct':peak_threshold,'metrics':metrics,'paired_comparisons':intervals,
        'versions':{'python':platform.python_version(),'torch':torch.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},
        'elapsed_seconds':time.perf_counter()-START}
(OUT/'results.json').write_text(json.dumps(result,indent=2))
np.savez_compressed(OUT/'test_predictions.npz',origin_times=data['test'][4].to_numpy().astype('datetime64[s]').astype('int64'),
                    truth=truth,persistence=baseline,trees=tree_pred,lstm_seed42=lstm_preds[0],
                    lstm_seed43=lstm_preds[1],lstm_seed44=lstm_preds[2],lstm_ensemble=predictions['LSTM ensemble'])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for name,color in [('Persistence','#888888'),('Boosted trees','#166b9b'),('LSTM seed 42','#c56624'),('LSTM ensemble','#6a4fa0')]:
    mm=[m for m in metrics if m['model']==name]
    axes[0].plot([10,30,60],[m['MAE_MW'] for m in mm],marker='o',label=name,color=color)
    axes[1].plot([10,30,60],[m['peak_MAE_MW'] for m in mm],marker='o',label=name,color=color)
for ax,title in zip(axes,['All held-out forecasts','High-demand forecasts']):
    ax.set(title=title,xlabel='Forecast horizon (minutes)',ylabel='Mean absolute error (MW)',xticks=[10,30,60]); ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
axes[0].legend(frameon=False,fontsize=9)
fig.suptitle('Total power forecasting: October–December 2023',fontsize=15)
fig.savefig(OUT/'comparison.png',dpi=170); plt.close(fig)

table=pd.DataFrame(metrics).round(4).to_html(index=False)
splits=pd.DataFrame(result['splits']).T.to_html()
findings='''<h2>What this experiment found</h2>
<p><strong>Persistence had the lowest mean absolute error at every horizon.</strong> Neither the small LSTM nor the selected boosted trees established an improvement on that primary metric. Trees beat every individual LSTM run at 30 and 60 minutes; at 10 minutes their errors were close.</p>
<p>Trees reduced root-mean-square error compared with persistence at all three horizons, including about 10% at 30 and 60 minutes. However, persistence also had lower absolute error on the high-demand subset. A model can reduce large squared errors overall while still underpredicting sustained peaks; the metrics answer different questions.</p>
<p>The three LSTM runs had 30-minute MAE between 1.279 and 1.415 MW and 60-minute MAE between 1.664 and 1.777 MW. Averaging their predictions improved on some individual runs but did not beat the baseline. The day-bootstrap intervals favor trees over seed 42 and the LSTM ensemble at 30 and 60 minutes; the seed-42 comparison at 10 minutes is inconclusive.</p>
<p>Practical choice from this benchmark: retain persistence for a simple MAE-focused forecast. Consider trees if reducing squared error is the priority. Further LSTM work needs evidence from broader chronological backtests, alternative lookback windows or workload/scheduler information.</p>'''
html=f'''<!doctype html><html><head><meta charset="utf-8"><title>Frontier forecasting experiment</title>
<style>body{{font:16px/1.5 system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#23303b}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}}th{{background:#f1f4f6}}img{{max-width:100%}}code{{background:#f1f4f6}}h1,h2{{line-height:1.2}}</style></head><body>
<h1>LSTM versus boosted trees for Frontier total power</h1>
<p>Target: total power 10, 30 and 60 minutes ahead. All models were evaluated on exactly the same held-out forecast origins. Lower errors are better. This is a modest benchmark on one chronological holdout, not a comprehensive hyperparameter search.</p>
{findings}
<img src="comparison.png" alt="Forecast error by horizon"><h2>Held-out results</h2>{table}
<p>Peak means actual target power at or above the training-period 95th percentile ({peak_threshold:.3f} MW). LSTM ensemble averages three independently initialized models and uses three times the individual-model inference effort. Seed 42 is the predeclared individual-model comparison; it was not selected on test performance.</p>
<h2>Data and validation</h2>{splits}
<p>Six-hour input history (36 observations); ten physical/power fields plus four calendar features at each time step. Trees receive the same sequence flattened. Both algorithms predict changes from current total power; persistence predicts no change. Input standardization uses January–July data only. Target windows stay within their assigned split. Historical context before validation/test boundaries is allowed because it is available at forecast time.</p>
<p>Timestamps within 1.01 seconds of minute boundaries are normalized. Data are reindexed to a complete 10-minute grid. No missing values are imputed; windows spanning any absent record are excluded. The error-prone average return-temperature field and calculated PUE/waste-heat fields are excluded. Other outliers and zero-flow records are retained. All input measurements are assumed available without reporting delay.</p>
<h2>Training choices</h2><p>LSTM: one 32-unit recurrent layer and three linear outputs; MAE loss, Adam learning rate 0.001, batch size 256, up to 30 epochs, early stopping after five validation epochs without improvement. Three seeds: 42, 43, 44. Each run restores its best validation checkpoint.</p>
<p>Trees: one histogram gradient-boosting regressor per horizon, 200 iterations, learning rate 0.05, MAE loss, 30 minimum samples per leaf and L2 penalty 1. Maximum leaf counts 15 and 31 were compared on validation mean MAE; selected value: {chosen_leaves}. No model was refit on validation data.</p>
<h2>Paired uncertainty checks</h2><pre>{json.dumps(intervals,indent=2)}</pre>
<p>Intervals resample entire calendar days together (2,000 paired bootstrap draws). Negative LSTM-minus-tree differences favor LSTM. They are approximate and do not capture dependence across days or uncertainty from other yearly operating regimes.</p>
<h2>Limits and reproducibility</h2><p>Only one year and one site; the final test quarter has no missing grid records, while earlier training periods do. No job scheduler, utilization, weather or maintenance labels. Model hyperparameter searches were deliberately small and are not matched in compute. This comparison establishes performance for this configuration and holdout, not universal superiority.</p>
<p>Source SHA-256: <code>{source_sha}</code>. Original workbook unchanged. Runtime versions: {result['versions']}. Total experiment time: {result['elapsed_seconds']/60:.1f} minutes.</p>
<p><a href="results.json">Detailed results</a> · <a href="run_experiment.py">Experiment code</a> · <a href="training_history.json">Training history</a></p>
</body></html>'''
(OUT/'report.html').write_text(html)
log('RESULTS',json.dumps(metrics,indent=2)); log('COMPLETE',result['elapsed_seconds'])
