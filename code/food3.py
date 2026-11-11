import pandas as pd, numpy as np, pycountry, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
f=pd.read_excel('data/Inflation-data.xlsx',sheet_name='fcpi_m')
f=f[f['Indicator Type'].astype(str).str.lower()=='index']
mcols=[c for c in f.columns if isinstance(c,(int,float)) and 197001<=c<=202612]
fx={'COD':'DR Congo (Zaire)','COG':'Congo','RUS':'Russia (Soviet Union)','MMR':'Myanmar (Burma)','YEM':'Yemen (North Yemen)','KHM':'Cambodia (Kampuchea)','TUR':'Turkey','BIH':'Bosnia-Herzegovina','SRB':'Serbia (Yugoslavia)','MKD':'Macedonia, FYR','ZWE':'Zimbabwe (Rhodesia)','SWZ':'Swaziland','TLS':'East Timor','CIV':'Ivory Coast','MDG':'Madagascar (Malagasy)','IRN':'Iran','SYR':'Syria','LAO':'Laos','BOL':'Bolivia','TZA':'Tanzania','VEN':'Venezuela','PSE':'Palestine','KOR':'South Korea','PRK':'North Korea','VNM':'Vietnam','MDA':'Moldova','GMB':'Gambia','KGZ':'Kyrgyzstan','CZE':'Czech Republic','USA':'United States of America','GBR':'United Kingdom','EGY':'Egypt'}
def nm(i):
    if i in fx: return fx[i]
    c=pycountry.countries.get(alpha_3=i); return c.name if c else i
rows=[]
for _,r in f.iterrows():
    c=nm(r['Country Code']); v=pd.to_numeric(r[mcols],errors='coerce'); v.index=[pd.Period(f'{int(m)//100}-{int(m)%100:02d}',freq='M') for m in mcols]
    chg=(np.log(v).diff(12)*100)
    for ym,x in chg.dropna().items(): rows.append((c,ym,x))
s=pd.DataFrame(rows,columns=['country','ym','chg12'])
s=s.groupby(['country','ym'],as_index=False).chg12.mean(); sp=s.set_index(['country','ym']).chg12
print('countries',s.country.nunique(),'country-months',len(s))
g=pd.read_csv('data/GEDEvent_v26_1.csv',usecols=['country','date_start','type_of_violence','best'])
g=g[g.type_of_violence==1]; g['ym']=pd.to_datetime(g.date_start).dt.to_period('M')
idx=pd.period_range('1989-01','2025-12',freq='M')
cm=g.groupby(['country','ym']).best.sum().unstack(fill_value=0).reindex(columns=idx,fill_value=0)
def find(quiet,thr12,first):
    out=[]
    for c in cm.index:
        v=cm.loc[c].values
        for t in range(24,len(v)-12):
            if v[t-24:t].max()<quiet and v[t:t+12].sum()>=thr12 and v[t]>=first: out.append((c,idx[t])); break
    return out
def lead(onsets,label):
    rows=[]
    for c,t in onsets:
        vals=[sp.get((c,t-k),np.nan) for k in range(1,7)]
        if np.all(np.isnan(vals)): continue
        # country's own baseline: median chg over the 5 years before the window
        own=[sp.get((c,t-k),np.nan) for k in range(7,67)]
        rows.append((c,str(t),np.nanmax(vals),np.nanmax(vals)-np.nanmedian(own)))
    r=pd.DataFrame(rows,columns=['country','onset','max_chg_6m','above_own_base'])
    base=s.chg12
    print(f'\n{label}: {len(onsets)} onsets, {len(r)} with food-CPI data before')
    print(f'  max 12-mo food-CPI rise in prior 6 months, median: onsets {r.max_chg_6m.median():.1f}%  all country-months {base.median():.1f}%')
    print(f'  above the country\'s own 5-year baseline, median: {r.above_own_base.median():.1f} points; share above +5: {np.mean(r.above_own_base>5):.2f}, above +10: {np.mean(r.above_own_base>10):.2f}')
    for thr in [10,20]: print(f'  share with rise > {thr}%: onsets {np.mean(r.max_chg_6m>thr):.2f}   base {np.mean(base>thr):.2f}')
    print(r.sort_values('max_chg_6m',ascending=False).round(1).to_string(index=False))
    return r
r1=lead(find(25,1000,25),'War onsets (1000 deaths within 12 months)')
r2=lead(find(5,300,25),'Conflict onsets (300 deaths within 12 months after 2 quiet years)')
# placebo: random quiet months in the same countries
rng=np.random.default_rng(0); pl=[]
for c in cm.index:
    v=cm.loc[c].values
    q=[t for t in range(24,len(v)-12) if v[t-24:t+12].max()<25]
    if q:
        for t in rng.choice(q,min(3,len(q)),replace=False): pl.append((c,idx[t]))
rp=lead(pl,'Placebo: random peaceful months')
# yearly panel
s['year']=s.ym.apply(lambda p:p.year)
ys=s.groupby(['country','year']).chg12.agg(['max','mean']).reset_index().rename(columns={'max':'fcpi_max','mean':'fcpi_mean'})
d=pd.read_csv('panels/panel2.csv').merge(ys,on=['country','year'],how='left')
d['fcpi_max']=d.fcpi_max.clip(-50,200)
print('\nyearly rows with food CPI:',d.fcpi_max.notna().sum(),'of',len(d))
inst=['v2x_polyarchy','v2x_rule','v2x_corr','v2x_freexp_altinf']
def run(target,feats,label):
    dd=d.dropna(subset=[target]+feats); tr=dd[dd.year<=2020]; te=dd[dd.year>=2021]
    lr=LogisticRegression(max_iter=5000).fit(tr[feats],tr[target])
    print(f'{label:40s} train n={len(tr)} on={int(tr[target].sum())} | test n={len(te)} on={int(te[target].sum())} | AUC {roc_auc_score(tr[target],lr.predict_proba(tr[feats])[:,1]):.3f} / {roc_auc_score(te[target],lr.predict_proba(te[feats])[:,1]):.3f}')
    return lr
dd=d.dropna(subset=['fcpi_max'])
for target in ['onset','onset1000']:
    x=dd.dropna(subset=[target]); print(f'{target}: peak food-CPI rise in year before, median: onset {x[x[target]==1].fcpi_max.median():.1f}%  none {x[x[target]==0].fcpi_max.median():.1f}%  (n onset={int(x[target].sum())})')
d=dd
run('onset',['since46','ever46']+inst,'onset: history + institutions')
lr=run('onset',['since46','ever46']+inst+['fcpi_max'],'onset: + food CPI peak rise'); print('  food coef',round(lr.coef_[0][-1],4))
run('onset1000',['since46','ever46','ldeaths','ldeaths_prev'],'war: history + deaths')
lr=run('onset1000',['since46','ever46','ldeaths','ldeaths_prev','fcpi_max'],'war: + food CPI peak rise'); print('  food coef',round(lr.coef_[0][-1],4))
d.to_csv('panels/panel3.csv',index=False)
