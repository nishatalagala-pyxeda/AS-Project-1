"""Independent saved-result verification and comparison report."""
import sys,os,json,hashlib,html
from pathlib import Path
sys.path.insert(0,'/private/tmp/frontier_experiment_deps')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/frontier_mpl_cache')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(os.environ.get('FRONTIER_FOREST_OUTPUT',str(Path(__file__).resolve().parent))).resolve()
parts=['''<!doctype html><html><head><meta charset="utf-8"><title>Frontier forest capacity sweep</title><style>
body{font:16px/1.55 system-ui,sans-serif;color:#172436;max-width:1200px;margin:45px auto;padding:0 25px}h1,h2,h3{line-height:1.2}table{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}th,td{text-align:right;padding:8px;border-bottom:1px solid #d7e0e8}th:first-child,td:first-child{text-align:left}img{width:100%}.note{background:#edf2f7;padding:18px}code{font-size:13px}a{color:#145da0}</style></head><body>
<h1>Forest accuracy and storage tradeoffs</h1>
<p>Random forests and Extra Trees for accessory power and calculated waste heat. Each family is tested at 1, 4, 16, 64, 128 and 256 trees; depth limits 2, 4, 6, 8 and 12; and two input sets: target history plus calendar, or all eleven available telemetry channels plus calendar. This gives <strong>120 configurations per target</strong>.</p>
<p>All forests use six hours of lag and summary features, minimum 10 samples per leaf, 70% features per split and seed 42. Both families share these settings; random forests bootstrap, while Extra Trees use their default non-bootstrap random splits. Five horizons are predicted jointly. This is a bounded capacity sweep, not an exhaustive search over histories, leaf sizes, feature fractions or seeds.</p>
<div class="note"><strong>Evaluation limits.</strong> October–December 2023 was inspected in earlier work. It is reused exploratory confirmation, not a fresh holdout. Model choices here use only three expanding chronological development score periods: June–July, August and September. Equal period weight is applied to MAE divided by persistence MAE. The source workbook is unchanged. Waste heat is the workbook’s calculated thermal quantity.</div>
<p>All configurations use identical origins with 24 hours of complete context and six hours of complete future data; no imputation. Training targets end before the next period. Inputs contain only observations available at the forecast origin. Forests learn residuals from the current target, with train-only target scaling and nonnegative final predictions. Source data and feature construction are reused from the earlier benchmark. Target-history-only waste heat still requires a calculated heat input derived from temperature and flow; it does not establish that the underlying sensors can be removed.</p>
<h2>Embedded model storage</h2><p>Storage is the actual exported array size: 16 bytes per internal node, 20 bytes per five-output leaf and 4 bytes per tree root. Coefficients and split thresholds are float32; split rounding preserves the original float32 feature comparisons. This excludes firmware, feature buffers, feature processing, metadata, stack, communications and calibration history. The budgets below are <strong>model-array budgets, not complete ESP32 memory budgets</strong>.</p>
<p>Each budget selects the lowest development error at each horizon among models whose maximum size across development periods fits the budget. The final fitted size is reported separately; any budget overrun is flagged without selecting a new winner using final-period errors. A different winner at each horizon would require storing multiple models; the joint-model table gives a single five-output alternative.</p>''']
all_checks=[];flat=[]
for target,title,mult,unit in [('accessory_power','Accessory power',1000,'kW'),('waste_heat','Waste heat',1,'MW')]:
    folder=ROOT/target;manifest=json.loads((folder/'manifest.json').read_text());ranking=json.loads((folder/'development_ranking.json').read_text());choices=json.loads((folder/'selection_before_confirmation.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    assert hashlib.sha256(Path(manifest['source']).read_bytes()).hexdigest()==manifest['source_sha256']
    assert len(ranking)==120 and len(choices)==20
    assert all(len(list((folder/f['name']).glob('*.json')))==120 for f in manifest['folds'][:-1])
    for item in ranking:
        records=[json.loads((folder/f['name']/(item['config']['id']+'.json')).read_text()) for f in manifest['folds'][:-1]]
        assert np.allclose(item['development_relative_MAE'],np.mean([r['metrics']['relative_MAE'] for r in records],axis=0),rtol=1e-12)
        assert item['maximum_development_packed_bytes']==max(r['size']['packed_model_bytes'] for r in records)
    for choice in choices:
        j=manifest['horizons_minutes'].index(choice['horizon_minutes']);budget=choice['budget_bytes']
        eligible=[r for r in ranking if budget is None or r['maximum_development_packed_bytes']<=budget]
        winner=min(eligible,key=lambda r:r['development_relative_MAE'][j]);assert winner['config']['id']==choice['model']
    joint=json.loads((folder/'joint_summary.json').read_text())
    selected=sorted({c['model'] for c in choices+joint})
    verified=0
    for name in selected:
        p=folder/'confirmation'/name;r=json.loads((folder/'confirmation'/(name+'.json')).read_text());z=np.load(p/'predictions.npz')
        assert z['truth'].shape==z['predictions'].shape==(13212,5)
        assert z['origin_unix_s'].min()>=int(np.datetime64('2023-10-01','s').astype('int64'))
        assert z['origin_unix_s'].max()+21600<int(np.datetime64('2024-01-01','s').astype('int64'))
        assert (np.diff(z['origin_unix_s'])>=600).all() and np.isfinite(z['predictions']).all() and (z['predictions']>=0).all()
        previous=np.load(ROOT.parent/'frontier_full_benchmark'/target/'confirmation'/'Persistence'/'predictions.npz')
        assert np.array_equal(z['origin_unix_s'],previous['origin_unix_s']) and np.array_equal(z['truth'],previous['truth'])
        e=z['predictions']-z['truth'];mae=np.abs(e).mean(0);base=np.abs(z['persistence']-z['truth']).mean(0)
        assert np.allclose(mae,r['metrics']['mae_MW'],rtol=1e-12)
        assert np.allclose(mae/base,r['metrics']['relative_MAE'],rtol=1e-12)
        assert np.allclose(np.sqrt((e*e).mean(0)),r['metrics']['rmse_MW'],rtol=1e-12)
        assert sum((p/f'{k}.bin').stat().st_size for k in ['nodes','leaves','roots'])==r['size']['packed_model_bytes']
        assert json.loads((p/'host_c_verification.json').read_text())['max_absolute_error_MW']<2e-4
        verified+=1
    all_checks.append(dict(target=target,development_evaluations=360,confirmation_models_verified=verified,selection_verified=True,source_unchanged=True))
    # Development-only single-model choices minimize the mean of the five relative errors.
    joint_checks=[]
    for budget in [None,32768,131072,524288]:
        eligible=[r for r in ranking if budget is None or r['maximum_development_packed_bytes']<=budget]
        winner=min(eligible,key=lambda r:np.mean(r['development_relative_MAE']))
        actual=next(r for r in joint if r['budget_bytes']==budget);assert actual['model']==winner['config']['id']
    # Scientific plots of all configs, not just selected configurations.
    fig,axs=plt.subplots(1,3,figsize=(15,4.7))
    for ax,j in zip(axs,[0,2,4]):
        for family,color in [('RF','#1565c0'),('ET','#c26412')]:
          for own,marker in [(True,'o'),(False,'^')]:
            subset=[r for r in ranking if r['config']['family']==family and r['config']['own']==own]
            ax.scatter([r['maximum_development_packed_bytes']/1024 for r in subset],[r['development_relative_MAE'][j] for r in subset],s=24,alpha=.7,c=color,marker=marker,label=f'{family}, {"target only" if own else "all telemetry"}')
        ax.axhline(1,color='#666',ls='--',lw=1);ax.set_xscale('log');ax.set_xlabel('Max. development model arrays (KiB)');ax.set_ylabel('Development MAE / persistence MAE');ax.set_title(f'{manifest["horizons_minutes"][j]} minute horizon');ax.grid(alpha=.15)
    axs[0].legend(fontsize=8);fig.suptitle(title+': all 120 forest configurations');fig.tight_layout();fig.savefig(folder/'tradeoff.png',dpi=160);plt.close(fig)
    parts += [f'<h2>{title}</h2><img src="{target}/tradeoff.png" alt="Development accuracy versus storage for all forest configurations">',f'<p>Lower error ratios are better; values below 1 beat persistence. Circles use target history, triangles use all telemetry. Storage is on a logarithmic scale. Complete metrics are in the <a href="{target}/development_ranking.json">development ranking</a>.</p>', '<h3>Selections at each forecast horizon</h3><table><tr><th>Model budget</th><th>Horizon</th><th>Selected forest</th><th>Final MAE ('+unit+')</th><th>Gain vs persistence</th><th>Final arrays (KiB)</th></tr>']
    for r in summary:
        label='Unrestricted' if r['budget_bytes'] is None else f'{r["budget_bytes"]/1024:g} KiB'
        flag='' if r['confirmation_budget_met'] else ' (over budget)'
        parts.append(f'<tr><td>{label}</td><td>{r["horizon_minutes"]} min</td><td><a href="{target}/confirmation/{r["model"]}/metadata.json">{r["model"]}</a></td><td>{r["confirmation_MAE_MW"]*mult:.3f}</td><td>{100*(1-r["confirmation_relative_MAE"]):.1f}%</td><td>{r["confirmation_packed_bytes"]/1024:.1f}{flag}</td></tr>')
        flat.append(dict(target=target,**r))
    intervals=[]
    for r in summary:
        z=np.load(folder/'confirmation'/r['model']/'predictions.npz');j=manifest['horizons_minutes'].index(r['horizon_minutes'])
        difference=np.abs(z['predictions'][:,j]-z['truth'][:,j])-np.abs(z['persistence'][:,j]-z['truth'][:,j])
        days=pd.factorize(pd.to_datetime(z['origin_unix_s'],unit='s').normalize())[0]
        counts=np.bincount(days);sums=np.bincount(days,weights=difference)
        draws=np.random.default_rng(2026).integers(0,len(counts),(2000,len(counts)))
        boot=sums[draws].sum(1)/counts[draws].sum(1)
        intervals.append(dict(budget_bytes=r['budget_bytes'],horizon_minutes=r['horizon_minutes'],model=r['model'],
            paired_MAE_difference_MW=float(difference.mean()),day_bootstrap_95pct_MW=np.quantile(boot,[.025,.975]).tolist()))
    (folder/'paired_intervals.json').write_text(json.dumps(intervals,indent=2))
    parts+=['</table><h3>One forest for all five horizons</h3><p>These choices minimize the mean development relative MAE across horizons. One selected forest predicts all five horizons, so its storage is shared. Sizes and horizon errors below come from the final fit.</p><table><tr><th>Budget</th><th>Forest</th><th>Mean development MAE / persistence</th><th>Final arrays (KiB)</th><th>Final MAE at 10 / 30 / 60 / 120 / 360 min ('+unit+')</th></tr>']
    for r in joint:
        errors=' / '.join(f'{e*mult:.3f}' for e in r['confirmation_mae_MW'])
        parts.append(f'<tr><td>{"Unrestricted" if r["budget_bytes"] is None else str(r["budget_bytes"]//1024)+" KiB"}</td><td>{r["model"]}</td><td>{r["development_mean_relative_MAE"]:.3f}</td><td>{r["confirmation_packed_bytes"]/1024:.1f}</td><td>{errors}</td></tr>')
    parts+=['</table>']
    # Compare with previous selected models on exactly the same confirmation origins.
    old=json.loads((ROOT.parent/'frontier_full_benchmark'/target/'summary.json').read_text())['selected_confirmation']
    parts+=['<h3>Comparison with the earlier benchmark</h3><table><tr><th>Horizon</th><th>Earlier selected model</th><th>Earlier MAE ('+unit+')</th><th>Unrestricted selected forest MAE ('+unit+')</th></tr>']
    for prev in old:
        r=next(r for r in summary if r['budget_bytes'] is None and r['horizon_minutes']==prev['horizon_minutes'])
        parts.append(f'<tr><td>{r["horizon_minutes"]} min</td><td>{prev["selected_model"]}</td><td>{prev["MAE_MW"]*mult:.3f}</td><td>{r["confirmation_MAE_MW"]*mult:.3f}</td></tr>')
    parts+=['</table>']
intro='<p><strong>Compact forests retain most of the forecasting accuracy of larger forests.</strong> The shared-model candidates below each predict all five horizons. Their 30-minute to six-hour errors improve on persistence; ten-minute forecasting offers no convincing benefit. The earlier boosted-tree and Ridge models remain competitive and often better.</p><table><tr><th>Compact shared model</th><th>Trees / depth</th><th>Arrays (KiB)</th><th>One-hour MAE</th><th>Gain vs persistence</th><th>Six-hour MAE</th><th>Gain vs persistence</th></tr>'
for target,title,mult,unit in [('accessory_power','Accessory power',1000,'kW'),('waste_heat','Waste heat',1,'MW')]:
    c=next(r for r in json.loads((ROOT/target/'joint_summary.json').read_text()) if r['budget_bytes']==32768)
    cfg=next(r for r in json.loads((ROOT/target/'manifest.json').read_text())['configs'] if r['id']==c['model'])
    intro+=f'<tr><td>{title} — Extra Trees</td><td>{cfg["trees"]} / {cfg["depth"]}</td><td>{c["confirmation_packed_bytes"]/1024:.1f}</td><td>{c["confirmation_mae_MW"][2]*mult:.3f} {unit}</td><td>{100*(1-c["confirmation_relative_MAE"][2]):.1f}%</td><td>{c["confirmation_mae_MW"][4]*mult:.3f} {unit}</td><td>{100*(1-c["confirmation_relative_MAE"][4]):.1f}%</td></tr>'
intro+='</table><p>These are exploratory results on the previously examined final period. Array sizes exclude firmware and working memory. The exported C inference agrees with Python forecasts on checked samples; ESP32 deployment and energy consumption have not been measured.</p>'
parts[0]=parts[0].replace('<h1>Forest accuracy and storage tradeoffs</h1>','<h1>Forest accuracy and storage tradeoffs</h1>'+intro)
parts+=['<h2>Export and validation</h2><p>Each final selected forest includes a Python model, raw packed arrays, feature names, target scaling, and compile-ready C arrays plus inference function. The C code is compiled and checked on the host against Python forecasts. It accepts an already prepared feature vector; sensor ingestion, rolling features, timekeeping, firmware integration and ESP32 measurements remain to be implemented. No on-device energy-saving claim is made.</p><p>All selected-model forecast metrics, all development rankings and selections, array sizes, source hashes and host C agreement were checked. Model capacity and feature-set selection were performed on development periods; the many tested configurations and previously inspected final period limit independent generalization claims. Daily block bootstrap comparisons with persistence are saved per target in paired_intervals.json; they are exploratory and do not correct for model search or earlier inspection of the final period.</p><p><a href="results.csv">Selected result table (CSV)</a> · <a href="verification.json">Verification details</a> · <a href="run_sweep.py">Experiment source</a> · <a href="../frontier_full_benchmark/report.html">Earlier full benchmark</a></p></body></html>']
(ROOT/'report.html').write_text('\n'.join(parts));pd.DataFrame(flat).to_csv(ROOT/'results.csv',index=False)
(ROOT/'verification.json').write_text(json.dumps(all_checks,indent=2));print(json.dumps(all_checks,indent=2))
