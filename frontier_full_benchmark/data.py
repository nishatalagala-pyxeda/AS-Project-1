"""Read-only extraction and leakage-aware feature construction."""
import hashlib,json,re,zipfile
import xml.etree.ElementTree as E
from pathlib import Path
import numpy as np
import pandas as pd

COLS=['B','C','D','E','F','G','I','O','P','Q','N']
HORIZONS=np.array([1,3,6,12,36])
FOLDS=[
    {'name':'June-July','train_end':'2023-05-01','validation_end':'2023-06-01','score_end':'2023-08-01'},
    {'name':'August','train_end':'2023-07-01','validation_end':'2023-08-01','score_end':'2023-09-01'},
    {'name':'September','train_end':'2023-08-01','validation_end':'2023-09-01','score_end':'2023-10-01'},
    {'name':'confirmation','train_end':'2023-09-01','validation_end':'2023-10-01','score_end':'2024-01-01'},
]

def load(source):
    source=Path(source);ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};rows=[]
    with zipfile.ZipFile(source) as z:
        for _,e in E.iterparse(z.open('xl/worksheets/sheet2.xml'),events=('end',)):
            if e.tag!='{'+ns['s']+'}row':continue
            if int(e.attrib['r'])>2:
                row={}
                for c in e:
                    v=c.find('s:v',ns)
                    if v is not None and c.attrib.get('t') not in ['s','e']:
                        row[re.match('[A-Z]+',c.attrib['r'])[0]]=float(v.text)
                if 'A' in row:rows.append(row)
            e.clear()
    d=pd.DataFrame(rows);t=pd.to_datetime(d.A,unit='D',origin='1899-12-30');rounded=t.dt.round('min')
    assert ((t-rounded).abs().dt.total_seconds()<=1.01).all()
    d.index=pd.DatetimeIndex(rounded);assert d.index.is_unique and d.index.is_monotonic_increasing
    grid=pd.date_range('2023-01-01','2023-12-31 23:50',freq='10min');d=d.reindex(grid)
    raw=d[COLS].to_numpy(np.float32);origins=np.arange(144,len(d)-36)
    bad=np.r_[0,np.cumsum(~np.isfinite(raw).all(1))]
    origins=origins[(bad[origins+37]-bad[origins-144])==0]
    hr=grid.hour.to_numpy()+grid.minute.to_numpy()/60;dow=grid.dayofweek.to_numpy()
    cal=np.column_stack([np.sin(2*np.pi*hr/24),np.cos(2*np.pi*hr/24),np.sin(2*np.pi*dow/7),np.cos(2*np.pi*dow/7)]).astype(np.float32)
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    return d,raw,grid,origins,cal,sha

def masks(grid,o,fold):
    t=grid[o];last=grid[o+36];a=pd.Timestamp(fold['train_end']);b=pd.Timestamp(fold['validation_end']);c=pd.Timestamp(fold['score_end'])
    mm={'train':last<a,'validation':(t>=a)&(last<b),'score':(t>=b)&(last<c)}
    assert all(m.sum()>100 for m in mm.values()),{k:int(v.sum()) for k,v in mm.items()}
    assert last[mm['train']].max()<t[mm['validation']].min()
    assert last[mm['validation']].max()<t[mm['score']].min()
    return mm

def features(raw,cal,o,history,target_index,own=False):
    indices=[target_index] if own else list(range(raw.shape[1]))
    v=raw[:,indices];lags=[k for k in [0,1,2,3,4,5,11,17,23,35,71,107,143] if k<history]
    chunks=[v[o-k] for k in lags];labels=[f'{COLS[c]}_lag_{k}' for k in lags for c in indices]
    for window in [6,18,36,72,144]:
        if window>history:continue
        values=np.stack([v[o-k] for k in range(window)],axis=1)
        chunks.extend([values.mean(1),values.std(1),values.min(1),values.max(1),values[:,0]-values[:,-1]])
        labels.extend([f'{COLS[c]}_{stat}_{window}' for stat in ['mean','std','min','max','change'] for c in indices])
    chunks.append(cal[o]);labels+=['hour_sin','hour_cos','weekday_sin','weekday_cos']
    x=np.column_stack(chunks).astype(np.float32);assert np.isfinite(x).all()
    return x,labels

def sequence(raw,cal,o,history,train_end,grid):
    fit=raw[grid<pd.Timestamp(train_end)];mu=np.nanmean(fit,0);sd=np.nanstd(fit,0);sd[sd==0]=1
    f=np.column_stack([(raw-mu)/sd,cal]).astype(np.float32)
    return np.stack([f[i-history+1:i+1] for i in o]),mu,sd
