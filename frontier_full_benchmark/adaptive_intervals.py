"""Exploratory causal rolling calibration for development-selected forecasts.

Both prespecified 7-day and 30-day windows are reported; neither is chosen by
confirmation performance. An error is added only after its target timestamp.
"""
import argparse,sys,json,bisect,math
from collections import deque
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('target',choices=['accessory_power','waste_heat']);p.add_argument('--deps',default='/private/tmp/frontier_experiment_deps');args=p.parse_args()
sys.path.insert(0,args.deps)
import numpy as np
import pandas as pd
from data import load,masks,FOLDS,HORIZONS,COLS
root=Path(__file__).resolve().parent/args.target
manifest=json.loads((root/'manifest.json').read_text());choices=json.loads((root/'champions_selected_before_confirmation.json').read_text())
d,raw,grid,o,cal,sha=load(manifest['source']);assert sha==manifest['source_sha256']
rows=[];maximum_future_error_violations=0
for fold in FOLDS:
    mm=masks(grid,o,fold);score_origin=grid[o[mm['score']]].to_numpy().astype('datetime64[s]').astype('int64')
    val_origin=grid[o[mm['validation']]].to_numpy().astype('datetime64[s]').astype('int64')
    for j,h in enumerate(HORIZONS*10):
        model=choices[str(h)]['model'];z=np.load(root/fold['name']/model/'predictions.npz')
        assert np.array_equal(score_origin,z['origin_unix_s'])
        assert len(val_origin)==len(z['validation_truth'])
        val_errors=np.abs(z['validation_predictions'][:,j]-z['validation_truth'][:,j])
        errors=np.abs(z['predictions'][:,j]-z['truth'][:,j]);pred=z['predictions'][:,j];truth=z['truth'][:,j]
        target_times=score_origin+int(h)*60;val_targets=val_origin+int(h)*60
        assert val_targets.max()<score_origin.min()
        static_radius=float(z['interval90_radius_MW'][j])
        for days in [7,30]:
            history=deque((int(t),float(e)) for t,e in zip(val_targets,val_errors));sorted_errors=sorted(float(e) for e in val_errors)
            pointer=0;radii=[];fallback=0;last_available=[]
            for i,now in enumerate(score_origin):
                while pointer<i and target_times[pointer]<=now:
                    history.append((int(target_times[pointer]),float(errors[pointer])));bisect.insort(sorted_errors,float(errors[pointer]));pointer+=1
                while history and history[0][0]<now-days*86400:
                    _,err=history.popleft();pos=bisect.bisect_left(sorted_errors,err);assert sorted_errors[pos]==err;sorted_errors.pop(pos)
                if history:
                    assert history[-1][0]<=now
                    n=len(sorted_errors);q=min(1,math.ceil((n+1)*.9)/n);rank=min(n-1,math.ceil(q*(n-1)))
                    radius=sorted_errors[rank];last_available.append(history[-1][0])
                else:radius=static_radius;fallback+=1;last_available.append(int(val_targets.max()))
                radii.append(radius)
            radii=np.array(radii);lower=np.maximum(0,pred-radii);upper=pred+radii
            coverage=float(np.mean((truth>=lower)&(truth<=upper)));width=float(np.mean(upper-lower))
            name='rolling_'+str(days)+'d'
            np.savez_compressed(root/fold['name']/model/(name+'_h'+str(h)+'.npz'),origin_unix_s=score_origin,radius_MW=radii,latest_used_error_target_unix_s=np.array(last_available))
            maximum_future_error_violations+=int(np.sum(np.array(last_available)>score_origin))
            assert np.all(np.array(last_available)<=score_origin)
            rows.append({'fold':fold['name'],'horizon_minutes':int(h),'model':model,'window_days':days,'coverage':coverage,'mean_width_MW':width,'n':len(truth),'fallback_n':fallback})
            print(fold['name'],h,days,'coverage',round(coverage,4),'width',round(width,4),flush=True)
note=('Exploratory diagnostic motivated by static accessory-power intervals under-covering confirmation. '+('Specified before waste-heat confirmation. ' if args.target=='waste_heat' else 'Added after seeing that undercoverage. ')+'Both 7-day and 30-day windows are reported without choosing one on confirmation. Only resolved forecast errors with target time <= current forecast origin are used.')
result={'note':note,'rows':rows,'future_error_violations':maximum_future_error_violations}
(root/'adaptive_intervals.json').write_text(json.dumps(result,indent=2))
manifest['adaptive_calibration_note']=note;(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Causal availability assertions passed for all windows.')
