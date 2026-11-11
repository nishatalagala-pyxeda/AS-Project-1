import zipfile, xml.etree.ElementTree as E, re, json, os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

path=os.environ.get('FRONTIER_SOURCE','/Users/nishatalagala/Downloads/Frontier HPC & Facility Data.xlsx')
z=zipfile.ZipFile(path)
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
ss=[''.join(n.itertext()) for n in E.fromstring(z.read('xl/sharedStrings.xml'))]
rows=[]; formulas=defaultdict(list); errors=[]; headers=None
for _,e in E.iterparse(z.open('xl/worksheets/sheet2.xml'),events=('end',)):
    if e.tag!='{'+ns['s']+'}row': continue
    rn=int(e.attrib['r']); vals=[None]*18
    for c in e:
        addr=c.attrib['r']; col=re.match('[A-Z]+',addr)[0]; idx=0
        for ch in col: idx=idx*26+ord(ch)-64
        if idx>18: continue
        v=c.find('s:v',ns); f=c.find('s:f',ns)
        value=v.text if v is not None else None
        if c.attrib.get('t')=='s': value=ss[int(value)]
        elif c.attrib.get('t')=='e': errors.append((addr,value))
        elif value is not None:
            try: value=float(value)
            except ValueError: pass
        vals[idx-1]=value
        if f is not None: formulas[col].append((addr,f.text,dict(f.attrib)))
    if rn==1: headers=vals
    elif rn>2 and any(v is not None for v in vals): rows.append((rn,vals))
    e.clear()
df=pd.DataFrame([v for _,v in rows],columns=list('ABCDEFGHIJKLMNOPQR'),index=[rn for rn,_ in rows])
print('SHAPE',df.shape,'ERRORS',len(errors),errors[:15])
print('MISSING',df.isna().sum().to_dict())
print('DTYPES',df.dtypes.astype(str).to_dict())
df=df.apply(pd.to_numeric,errors='coerce')
print('FORMULAS', {k:{'count':len(v),'first':v[:3],'last':v[-2:]} for k,v in formulas.items()})
t=pd.to_datetime(df.A,unit='D',origin='1899-12-30').dt.round('s')
print('TIME',str(t.min()),str(t.max()),'duplicate',t.duplicated().sum(),'out_of_order',(t.diff().dt.total_seconds()<0).sum(),'steps',t.diff().dt.total_seconds().value_counts().to_dict())
print('DUPLICATE_ROWS',df.duplicated().sum())
print('STATS',df.iloc[:,1:].describe(percentiles=[.01,.5,.99]).to_json())
print('ZERO_NEG',{c:{'zero':int((df[c]==0).sum()),'negative':int((df[c]<0).sum()),'unique':int(df[c].nunique())} for c in df.columns[1:]})
def check(name,a,b,tol=1e-7):
    diff=(a-b).abs(); bad=diff>tol
    print('CHECK',name,'mismatch',int(bad.sum()),'max_diff',diff.max(),'examples',df.loc[bad].head(3).to_dict('index'))
check('flow sum',df.J,df.C+df.E+df.G)
check('total power',df.Q,df.O+df.P)
check('waste sum',df.N,df.K+df.L+df.M)
check('PUE',df.R,df.Q/df.O)
check('weighted return temp',df.H,(df.B*df.C+df.D*df.E+df.F*df.G)/df.J)
for temp,flow,heat in [('B','C','K'),('D','E','L'),('F','G','M')]:
    check('heat '+heat,df[heat],np.maximum(df[temp]-df.I,0)*df[flow]*(1060*3.5*3.785/60/1e6))
    mask=df[temp]<df.I
    print('RETURN_BELOW_SUPPLY',temp,int(mask.sum()),'sample rows',df.index[mask].tolist()[:10])
print('PUE_LT1',int((df.R<1).sum()),'HEAT_GT_COMPUTE',int((df.N>df.O).sum()),'HEAT_GT_TOTAL',int((df.N>df.Q).sum()))
print('ENERGY assuming 10min average', {c:float(df[c].sum()/6) for c in ['O','P','Q','N']},'weighted_pue',df.Q.sum()/df.O.sum(),'mean_pue',df.R.mean())
monthly=df.assign(month=t.dt.strftime('%Y-%m')).groupby('month').agg(n=('O','size'),compute=('O','mean'),total=('Q','mean'),heat=('N','mean'),pue=('R','mean'),supply=('I','mean'))
print('MONTHLY',monthly.to_json(orient='index'))
print('CONSTANT_RUNS')
for c in ['B','C','D','E','F','G','I','O','P','Q']:
    groups=(df[c]!=df[c].shift()).cumsum(); counts=df.groupby(groups).size(); g=counts.idxmax(); ix=df.index[groups==g]
    print(c,'longest',int(counts.max()),'start',str(t.loc[ix[0]]),'end',str(t.loc[ix[-1]]),'value',df.loc[ix[0],c])
print('CORR',df[['N','O','P','Q','I']].corr().round(3).to_json())
print('CHARTS')
for name in z.namelist():
    if re.match(r'xl/charts/chart\d+\.xml$',name):
        root=E.fromstring(z.read(name)); cn={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
        print(name,[x.text for x in root.findall('.//c:f',cn)])
df.assign(timestamp=t).to_pickle('/private/tmp/frontier_assessment.pkl')
