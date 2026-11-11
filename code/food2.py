import pandas as pd, numpy as np, pycountry, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
s=pd.read_csv('panels/staple_chg.csv'); s['ym']=pd.PeriodIndex(s.ym,freq='M')
fix={'COD':'DR Congo (Zaire)','COG':'Congo','RUS':'Russia (Soviet Union)','MMR':'Myanmar (Burma)','YEM':'Yemen (North Yemen)','KHM':'Cambodia (Kampuchea)','TUR':'Turkey','BIH':'Bosnia-Herzegovina','SRB':'Serbia (Yugoslavia)','MKD':'Macedonia, FYR','ZWE':'Zimbabwe (Rhodesia)','SWZ':'Swaziland','TLS':'East Timor','CIV':'Ivory Coast','MDG':'Madagascar (Malagasy)','IRN':'Iran','SYR':'Syria','LAO':'Laos','BOL':'Bolivia','TZA':'Tanzania','VEN':'Venezuela','PSE':'Palestine','KOR':'South Korea','PRK':'North Korea','VNM':'Vietnam','MDA':'Moldova','GMB':'Gambia','CPV':'Cape Verde','KGZ':'Kyrgyzstan'}
def nm(i):
    if i in fix: return fix[i]
    c=pycountry.countries.get(alpha_3=i); return c.name if c else i
s['country']=s.countryiso3.map(nm)
# ---- monthly: war onsets from GED (recompute)
g=pd.read_csv('data/GEDEvent_v26_1.csv',usecols=['country','date_start','type_of_violence','best'])
g=g[g.type_of_violence==1]; g['ym']=pd.to_datetime(g.date_start).dt.to_period('M')
idx=pd.period_range('1989-01','2025-12',freq='M')
cm=g.groupby(['country','ym']).best.sum().unstack(fill_value=0).reindex(columns=idx,fill_value=0)
onsets=[]
for c in cm.index:
    v=cm.loc[c].values
    for t in range(24,len(v)-12):
        if v[t-24:t].max()<25 and v[t:t+12].sum()>=1000 and v[t]>=25: onsets.append((c,idx[t])); break
# also lower threshold: 25+ in a month after 24 quiet months (<5/month), 300+ in next 12
onsets25=[]
for c in cm.index:
    v=cm.loc[c].values
    for t in range(24,len(v)-12):
        if v[t-24:t].max()<5 and v[t:t+12].sum()>=300 and v[t]>=25: onsets25.append((c,idx[t])); break
sp=s.set_index(['country','ym']).chg12
def lead(onsets,label):
    rows=[]
    for c,t in onsets:
        vals=[sp.get((c,t-k),np.nan) for k in range(1,7)]
        if np.all(np.isnan(vals)): continue
        rows.append((c,str(t),np.nanmax(vals)))
    r=pd.DataFrame(rows,columns=['country','onset','max_chg_6m'])
    base=s.chg12
    print(f'\n{label}: {len(onsets)} onsets, {len(r)} with local price data in the 6 months before')
    print(f'  median of max 12-mo staple rise in prior 6 months: onsets {r.max_chg_6m.median():.1f}%   all country-months {base.median():.1f}%')
    for thr in [10,20,30]:
        print(f'  share with rise > {thr}%: onsets {np.mean(r.max_chg_6m>thr):.2f}   base {np.mean(base>thr):.2f}')
    print(r.sort_values('max_chg_6m',ascending=False).to_string(index=False))
lead(onsets,'War onsets (1000 deaths in 12 months)')
lead(onsets25,'Conflict onsets (300 deaths in 12 months after 2 quiet years)')
# ---- yearly panel: add max staple change in year t
s['year']=s.ym.dt.year
ys=s.groupby(['country','year']).chg12.agg(['max','mean']).reset_index().rename(columns={'max':'stap_max','mean':'stap_mean'})
d=pd.read_csv('panels/panel2.csv').merge(ys,on=['country','year'],how='left')
print('\nrows with local staple data:',d.stap_max.notna().sum(),'of',len(d))
inst=['v2x_polyarchy','v2x_rule','v2x_corr','v2x_freexp_altinf']
def run(target,feats,label):
    dd=d.dropna(subset=[target]+feats); tr=dd[dd.year<=2020]; te=dd[dd.year>=2021]
    lr=LogisticRegression(max_iter=3000).fit(tr[feats],tr[target])
    print(f'{label:36s} train n={len(tr)} on={int(tr[target].sum())} | test n={len(te)} on={int(te[target].sum())} | AUC {roc_auc_score(tr[target],lr.predict_proba(tr[feats])[:,1]):.3f} / {roc_auc_score(te[target],lr.predict_proba(te[feats])[:,1]):.3f}  coef stap={lr.coef_[0][-1]:.3f}')
print('== within countries that have WFP price data')
run('onset',['since46','ever46']+inst+['stap_mean'],'onset: history+inst+staple (placebo=0)') if False else None
dd=d[d.stap_max.notna()]
for target in ['onset','onset1000']:
    x=dd.dropna(subset=[target]+inst)
    a_=x[x[target]==1].stap_max.median(); b_=x[x[target]==0].stap_max.median()
    print(f'{target}: max staple rise in year before, median: onset {a_:.1f}%  none {b_:.1f}%  (n onset={int(x[target].sum())})')
d0=d.copy(); d=dd
run('onset',['since46','ever46']+inst,'onset: history + institutions')
run('onset',['since46','ever46']+inst+['stap_max'],'onset: + local staple max rise')
run('onset1000',['since46','ever46','ldeaths','ldeaths_prev'],'war: history + deaths')
run('onset1000',['since46','ever46','ldeaths','ldeaths_prev','stap_max'],'war: + local staple max rise')
