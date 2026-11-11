"""Repeat the predeclared waste-heat architecture at seeds 43 and 44.
Run after run_targets.py --task all, using the same Python/dependencies.
The primary seed42 experiment and all tree predictions remain unchanged.
"""
import json
import run_targets as ex
np=ex.np
pd=ex.pd

d,grid,o,X,masks,meta=ex.load_data()
v=d.N.to_numpy(np.float32);h=np.array([1,3,6]);y=v[o[:,None]+h]
b=np.repeat(v[o,None],3,axis=1)
scale=float(np.nanstd(v[grid<pd.Timestamp('2023-08-01')]))
residual=((y-b)/scale).astype(np.float32);te=masks['test']
directory=ex.OUT/'waste_heat';r=json.loads((directory/'results.json').read_text())
r['additional_lstm_runs']=[]
for seed in [43,44]:
    sub=directory/f'seed_{seed}';sub.mkdir(exist_ok=True)
    vp,lp,training=ex.train_lstm(X,residual,masks,sub,seed=seed)
    pred=lp*scale+b[te];truth=y[te];metrics=[];cis=[]
    for j,step in enumerate(h):
        err=pred[:,j]-truth[:,j]
        metrics.append({'horizon_minutes':int(step*10),'MAE':float(np.abs(err).mean()),'RMSE':float(np.sqrt(np.mean(err**2))),
                        'improvement_vs_persistence_pct':float(100*(1-np.abs(err).mean()/np.abs(b[te,j]-truth[:,j]).mean()))})
        cis.extend([dict(horizon_minutes=int(step*10),**ci) for ci in ex.paired_ci({'LSTM':np.abs(err)},grid[o[te]],np.abs(b[te,j]-truth[:,j]))])
    np.savez_compressed(sub/'predictions.npz',lstm=pred,truth=truth,persistence=b[te],origin_unix_s=grid[o[te]].to_numpy().astype('datetime64[s]').astype('int64'))
    r['additional_lstm_runs'].append({'seed':seed,'training':training,'metrics':metrics,'paired_vs_persistence':cis})
    ex.save_json(directory/'results.json',r)
    ex.log('REPLICATION',seed,metrics)
ex.report();ex.log('REPLICATIONS COMPLETE')
