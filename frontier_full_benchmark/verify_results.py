"""Independent verification of saved forecasts, intervals and model selection."""
import argparse,sys,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('target',choices=['accessory_power','waste_heat']);p.add_argument('--deps',default='/private/tmp/frontier_experiment_deps');args=p.parse_args()
sys.path.insert(0,args.deps)
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parent/args.target
summary=json.loads((root/'summary.json').read_text());manifest=summary['manifest']
assert hashlib.sha256(Path(manifest['source']).read_bytes()).hexdigest()==manifest['source_sha256']
expected_models={c['id'] for c in manifest['configs']}|{'Persistence','Trailing median','Daily pattern'}
folds={f['name']:f for f in manifest['folds']};fingerprints={};verified=0
for r in summary['all_results']:
    fold=r['fold'];directory=root/fold['name']/r['model'];z=np.load(directory/'predictions.npz')
    y=z['truth'];pred=z['predictions'];base=z['persistence'];times=pd.to_datetime(z['origin_unix_s'],unit='s')
    assert y.shape==pred.shape==base.shape and y.shape[1]==5
    assert all(np.isfinite(z[k]).all() for k in z.files)
    assert (pred>=0).all() and times.min()>=pd.Timestamp(fold['validation_end'])
    assert int(z['origin_unix_s'].max())+21600<int(np.datetime64(fold['score_end'],'s').astype('int64'))
    assert (np.diff(z['origin_unix_s'])>=600).all()
    fingerprint=hashlib.sha256(y.tobytes()+z['origin_unix_s'].tobytes()).hexdigest()
    if fold['name'] in fingerprints:assert fingerprint==fingerprints[fold['name']]
    else:fingerprints[fold['name']]=fingerprint
    n=len(z['validation_truth']);level=min(1,np.ceil((n+1)*.9)/n)
    radius=np.quantile(np.abs(z['validation_truth']-z['validation_predictions']),level,axis=0,method='higher')
    assert np.allclose(radius,z['interval90_radius_MW'])
    for j,m in enumerate(r['metrics']):
        e=pred[:,j]-y[:,j]
        assert np.isclose(np.abs(e).mean(),m['MAE_MW'])
        assert np.isclose(np.sqrt(np.mean(e**2)),m['RMSE_MW'])
        assert np.isclose(e.mean(),m['bias_MW'])
        assert np.isclose(np.abs(e).mean()/np.abs(base[:,j]-y[:,j]).mean(),m['MAE_relative_to_persistence'])
        cover=np.mean((y[:,j]>=np.maximum(0,pred[:,j]-radius[j]))&(y[:,j]<=pred[:,j]+radius[j]))
        assert np.isclose(cover,m['interval90_coverage'])
        peak=y[:,j]>=r['peak_threshold_MW'];assert peak.sum()==m['peak_n']
        if peak.any():assert np.isclose(np.abs(e[peak]).mean(),m['peak_MAE_MW'])
    verified+=1
for fold in folds:
    assert {r['model'] for r in summary['all_results'] if r['fold']['name']==fold}==expected_models
for r in summary['neural_replications']:
    seed=r['training']['seed'];assert seed in [43,44]
    z=np.load(root/'confirmation'/r['model']/('seed'+str(seed))/'predictions.npz')
    fingerprint=hashlib.sha256(z['truth'].tobytes()+z['origin_unix_s'].tobytes()).hexdigest()
    assert fingerprint==fingerprints['confirmation']
    for j,m in enumerate(r['metrics']):
        e=z['predictions'][:,j]-z['truth'][:,j]
        assert np.isclose(np.abs(e).mean(),m['MAE_MW'])
        assert np.isclose(np.sqrt(np.mean(e**2)),m['RMSE_MW'])
for h in manifest['horizons_minutes']:
    rankings=[r for r in summary['development_ranking'] if r['horizon_minutes']==h]
    for rank in rankings:
        vals=[next(m['MAE_relative_to_persistence'] for m in r['metrics'] if m['horizon_minutes']==h) for r in summary['all_results'] if r['model']==rank['model'] and r['fold']['name']!='confirmation']
        assert len(vals)==3 and np.isclose(np.mean(vals),rank['development_mean_relative_MAE'])
    best=min(rankings,key=lambda r:r['development_mean_relative_MAE'])
    actual=next(s for s in summary['selected_confirmation'] if s['horizon_minutes']==h)
    assert best['model']==actual['selected_model']
assert len(fingerprints)==4
if summary.get('adaptive_calibration'):
    from data import load,masks
    _,_,grid,o,_,sha=load(manifest['source']);assert sha==manifest['source_sha256']
    for row in summary['adaptive_calibration']['rows']:
        fold=folds[row['fold']];mm=masks(grid,o,fold);h=row['horizon_minutes'];j=manifest['horizons_minutes'].index(h);days=row['window_days']
        directory=root/row['fold']/row['model'];z=np.load(directory/'predictions.npz');band=np.load(directory/('rolling_'+str(days)+'d_h'+str(h)+'.npz'))
        now=z['origin_unix_s'];assert np.array_equal(now,band['origin_unix_s']);assert (band['latest_used_error_target_unix_s']<=now).all()
        radius=band['radius_MW'];pred=z['predictions'][:,j];truth=z['truth'][:,j]
        cover=np.mean((truth>=np.maximum(0,pred-radius))&(truth<=pred+radius))
        assert np.isclose(cover,row['coverage'])
        assert np.isclose(np.mean(pred+radius-np.maximum(0,pred-radius)),row['mean_width_MW'])
        vt=grid[o[mm['validation']]].to_numpy().astype('datetime64[s]').astype('int64')+h*60
        ve=np.abs(z['validation_predictions'][:,j]-z['validation_truth'][:,j]);st=now+h*60;se=np.abs(pred-truth)
        for i in sorted(set([0,1,6,36,144,len(now)//2,len(now)-1])):
            if i>=len(now):continue
            past=np.r_[ve[(vt<=now[i])&(vt>=now[i]-days*86400)],se[(np.arange(len(now))<i)&(st<=now[i])&(st>=now[i]-days*86400)]]
            if len(past):
                n=len(past);q=min(1,np.ceil((n+1)*.9)/n);expected=np.quantile(past,q,method='higher')
            else:expected=z['interval90_radius_MW'][j]
            assert np.isclose(expected,radius[i]),(row['fold'],h,days,i)
    print('VERIFIED rolling-band coverage, widths and sampled radius calculations using only resolved errors.')
print('VERIFIED',args.target,verified,'model-period evaluations; common samples, metrics, train/validation/score boundaries, interval calibration, source SHA and development-only selection.')
print('SELECTED',json.dumps(summary['selected_confirmation'],indent=2))
