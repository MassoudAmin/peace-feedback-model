import pandas as pd, numpy as np, glob, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
files=sorted(glob.glob('data/[eE]vents*20*.tab')+glob.glob('data/events_20*.tsv'))
parts=[]
for f in files:
    d=pd.read_csv(f,sep='\t',usecols=['Event Date','CAMEO Code','Intensity','Country','Source Country','Target Country'],low_memory=False,quoting=3,on_bad_lines='skip')
    parts.append(d)
d=pd.concat(parts); d['ym']=pd.to_datetime(d['Event Date'],errors='coerce').dt.to_period('M'); d=d.dropna(subset=['ym'])
d['root']=pd.to_numeric(d['CAMEO Code'].astype(str).str.zfill(3).str[:2],errors='coerce'); d=d.dropna(subset=['root'])
d['hostile']=(d.root>=14).astype(int); d['verbal']=d.root.between(9,13).astype(int)
fx={'Democratic Republic of Congo':'DR Congo (Zaire)','Russian Federation':'Russia (Soviet Union)','Myanmar':'Myanmar (Burma)','Yemen':'Yemen (North Yemen)','Cambodia':'Cambodia (Kampuchea)','Bosnia and Herzegovina':'Bosnia-Herzegovina','Serbia':'Serbia (Yugoslavia)','Zimbabwe':'Zimbabwe (Rhodesia)',"Côte d'Ivoire":'Ivory Coast','Madagascar':'Madagascar (Malagasy)','United States':'United States of America','Occupied Palestinian Territory':'Palestine','Republic of Congo':'Congo','Macedonia':'Macedonia, FYR','Viet Nam':'Vietnam','Iran':'Iran','Syria':'Syria','Libya':'Libya','Egypt':'Egypt'}
m=d.dropna(subset=['Country']).groupby(['Country','ym']).agg(n=('Intensity','size'),intensity=('Intensity','mean'),hostile=('hostile','mean'),verbal=('verbal','mean')).reset_index()
m=m[m.n>=20]; m['country']=m.Country.map(lambda c: fx.get(c,c))
m.to_csv('panels/tone_all.csv',index=False)
print('years:',sorted(m.ym.dt.year.unique())); print('country-months',len(m))
# monthly series for onset test
sp=m.set_index(['country','ym'])
g=pd.read_csv('data/GEDEvent_v26_1.csv',usecols=['country','date_start','type_of_violence','best'])
g=g[g.type_of_violence==1]; g['ym']=pd.to_datetime(g.date_start).dt.to_period('M')
idx=pd.period_range('1989-01','2025-12',freq='M')
cm=g.groupby(['country','ym']).best.sum().unstack(fill_value=0).reindex(columns=idx,fill_value=0)
def find(quiet,thr12,first):
    out=[]
    for c in cm.index:
        v=cm.loc[c].values
        for t in range(24,len(v)-12):
            if v[t-24:t].max()<quiet and v[t:t+12].sum()>=thr12 and v[t]>=first: out.append((c,idx[t]))
    return out
def lead(onsets,label,var):
    rows=[]
    for c,t in onsets:
        pre=[sp[var].get((c,t-k),np.nan) for k in range(1,7)]      # months -6..-1
        base=[sp[var].get((c,t-k),np.nan) for k in range(7,25)]    # months -24..-7
        if np.sum(~np.isnan(pre))<3 or np.sum(~np.isnan(base))<6: continue
        rows.append((c,str(t),np.nanmean(pre)-np.nanmean(base)))
    r=pd.DataFrame(rows,columns=['country','onset','rise'])
    # placebo: same statistic at random quiet months
    rng=np.random.default_rng(1); pl=[]
    for c in cm.index:
        v=cm.loc[c].values; q=[t for t in range(24,len(v)-12) if v[t-24:t+12].max()<25]
        for t in rng.choice(q,min(4,len(q)),replace=False) if q else []: 
            pre=[sp[var].get((c,idx[t]-k),np.nan) for k in range(1,7)]; base=[sp[var].get((c,idx[t]-k),np.nan) for k in range(7,25)]
            if np.sum(~np.isnan(pre))>=3 and np.sum(~np.isnan(base))>=6: pl.append(np.nanmean(pre)-np.nanmean(base))
    pl=np.array(pl)
    print(f'\n{label} [{var}]: {len(r)} onsets with news data; placebo n={len(pl)}')
    print(f'  rise in 6 months before onset vs prior 18 months: onsets median {r.rise.median():+.3f}, placebo median {np.median(pl):+.3f}; share of onsets above placebo 75th pct: {np.mean(r.rise>np.percentile(pl,75)):.2f} (chance 0.25)')
    return r
w=find(25,1000,25); c25=find(5,300,25)
for var in ['hostile','intensity']:
    r=lead(w,'War onsets',var); print(r.sort_values('rise',ascending=False).round(3).to_string(index=False))
    lead(c25,'Conflict onsets',var)
# yearly panel: annual mean hostile share and its change
m['year']=m.ym.dt.year
ya=m.groupby(['country','year']).agg(host=('hostile','mean'),inten=('intensity','mean'),nn=('n','sum')).reset_index()
ya=ya.sort_values(['country','year']); ya['host_d']=ya.groupby('country').host.diff()
p=pd.read_csv('panels/panel4.csv').merge(ya,on=['country','year'],how='left')
inst=['v2x_polyarchy','v2x_rule','v2x_corr','v2x_freexp_altinf']
pp=p.dropna(subset=['host'])
print('\nyearly rows with news tone:',len(pp))
for target in ['onset','onset1000']:
    x=pp.dropna(subset=[target]); print(f'{target}: hostile share in year before, median: onset {x[x[target]==1].host.median():.3f} none {x[x[target]==0].host.median():.3f}; change vs prior year: onset {x[x[target]==1].host_d.median():+.3f} none {x[x[target]==0].host_d.median():+.3f} (n onset={int(x[target].sum())})')
def run(target,feats,label):
    dd=pp.dropna(subset=[target]+feats); tr=dd[dd.year<=2020]; te=dd[dd.year>=2021]
    if te[target].sum()<3: print(label,'too few test onsets'); return
    lr=LogisticRegression(max_iter=5000).fit(tr[feats],tr[target])
    print(f'{label:40s} train n={len(tr)} on={int(tr[target].sum())} | test n={len(te)} on={int(te[target].sum())} | AUC {roc_auc_score(tr[target],lr.predict_proba(tr[feats])[:,1]):.3f} / {roc_auc_score(te[target],lr.predict_proba(te[feats])[:,1]):.3f}')
run('onset',['since46','ever46']+inst,'onset: history + institutions')
run('onset',['since46','ever46']+inst+['host','host_d'],'onset: + news hostility')
run('onset1000',['since46','ever46','ldeaths','ldeaths_prev'],'war: history + deaths')
run('onset1000',['since46','ever46','ldeaths','ldeaths_prev','host','host_d'],'war: + news hostility')
p.to_csv('panels/panel5.csv',index=False)
