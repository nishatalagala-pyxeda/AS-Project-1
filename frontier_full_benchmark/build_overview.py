"""Build a compact overview of completed, independently verified targets."""
import sys,json,html
from pathlib import Path
sys.path.insert(0,'/private/tmp/frontier_experiment_deps')
import pandas as pd
root=Path(__file__).resolve().parent
parts=['''<!doctype html><html><head><meta charset="utf-8"><title>Frontier forecasting benchmark</title><style>body{font:16px/1.5 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#23303b}table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef2f5}th:first-child,td:first-child{text-align:left}img{max-width:100%}.note{padding:15px;background:#f3f5f7}code{overflow-wrap:anywhere}</style></head><body><h1>Accessory-power and waste-heat forecasting</h1>
<p>Broad benchmark:15 learned model configurations and three baselines, five forecast horizons, histories from one to24 hours, three chronological development backtests, and reused October–December confirmation. Accessory power was completed first, followed by waste heat.</p>
<p class="note">October–December was examined during earlier exploration. These are exploratory confirmation results, not performance on a new independent lockbox. Each horizon's model was selected using mean relative MAE in June–July, August and September, before this run scored confirmation.</p>''']
complete=[]
for target,label,scale,unit in [('accessory_power','Accessory power',1000,'kW'),('waste_heat','Waste heat',1,'MW')]:
    path=root/target/'summary.json'
    if not path.exists():continue
    r=json.loads(path.read_text());complete.append(target);rows=[]
    lookup={c['id']:c['family']+' '+str(c['history']//6)+'h'+(' target-only' if c.get('own') else '') for c in r['manifest']['configs']}
    for s in r['selected_confirmation']:
        ci=next(c for c in r['selected_paired_intervals'] if c['horizon_minutes']==s['horizon_minutes'])
        rows.append({'Horizon (min)':s['horizon_minutes'],'Selected model':lookup.get(s['selected_model'],s['selected_model']),
                     'MAE ('+unit+')':round(s['MAE_MW']*scale,3),'Gain vs persistence (%)':round(100*(1-s['MAE_relative_to_persistence']),1),
                     'Daily MAE difference 95% interval ('+unit+')':'['+', '.join(f'{v*scale:.3f}' for v in ci['day_bootstrap_95pct'])+']'})
    parts.append('<h2>'+label+'</h2>'+pd.DataFrame(rows).to_html(index=False)+'<p>Positive gain means lower absolute error than holding the current value unchanged. A difference interval entirely below zero favors the selected forecast, conditional on this test period.</p><img src="'+target+'/comparison.png" alt="'+label+' comparison"><p><a href="'+target+'/report.html">Full '+label.lower()+' report:all candidates, development stability, telemetry ablations, peak errors, interval widths and seeds</a></p>')
    selected60=next(s for s in r['selected_confirmation'] if s['horizon_minutes']==60)
    baseline60=next(m for result in r['all_results'] if result['fold']['name']=='confirmation' and result['model']=='Persistence' for m in result['metrics'] if m['horizon_minutes']==60)
    if selected60['peak_MAE_MW'] is not None and baseline60['peak_MAE_MW'] is not None:
        parts.append(f'<p>At one hour, high-demand MAE is {selected60["peak_MAE_MW"]*scale:.3f} {unit} for the selected model versus {baseline60["peak_MAE_MW"]*scale:.3f} {unit} for persistence. Overall gains and performance during peaks are separate findings.</p>')
    if r.get('adaptive_calibration'):
        aa=r['adaptive_calibration'];bands=[]
        for s in r['selected_confirmation']:
            h=s['horizon_minutes'];rr=[x for x in aa['rows'] if x['fold']=='confirmation' and x['horizon_minutes']==h]
            bands.append({'Horizon (min)':h,'Static coverage (%)':round(s['interval90_coverage']*100,1),
                          '7-day coverage (%)':round(next(x['coverage'] for x in rr if x['window_days']==7)*100,1),
                          '30-day coverage (%)':round(next(x['coverage'] for x in rr if x['window_days']==30)*100,1),
                          '7-day width ('+unit+')':round(next(x['mean_width_MW'] for x in rr if x['window_days']==7)*scale,2)})
        parts.append('<h3>Nominal 90% prediction intervals</h3>'+pd.DataFrame(bands).to_html(index=False)+'<p>'+html.escape(aa['note'])+'</p>')
parts.append('<h2>Reading the results</h2><p>90% coverage is the fraction of actual readings inside the forecast intervals. Static bands use validation errors; rolling bands use only already-resolved errors. Coverage substantially below90% indicates that those uncertainty bands did not transfer reliably. Point-error improvements and uncertainty calibration are separate findings.</p><p>All models use the same complete-history forecast origins. This removes data around source gaps, including many May/June records. Fixed settings give broad coverage of model families rather than exhaustive tuning. Independent verification recalculates saved metrics and interval coverage, checks common timestamps and source hashes, and confirms selection used development scores.</p><p>Only already-observed telemetry is used. Original workbook unchanged. Waste heat remains a calculated estimate with the source’s fixed thermal constants and zero-clipping assumptions. No operating changes or causal efficiency claims are implied.</p><p><a href="run_benchmark.py">Benchmark code</a> · <a href="data.py">Data preparation</a> · <a href="reproduce.txt">Reproduction instructions</a></p></body></html>')
(root/'report.html').write_text(''.join(parts))
print('Overview built for',complete)
