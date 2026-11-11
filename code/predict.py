import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
d=pd.read_csv('panels/panel2.csv')
inst=['v2x_polyarchy','v2x_rule','v2x_corr','v2x_freexp_altinf']
fo=['since46','ever46']+inst+['mil']; fw=['since46','ever46','ldeaths','ldeaths_prev']
# fill mil forward within country for 2025 where missing
d=d.sort_values(['country','year']); d['mil']=d.groupby('country').mil.ffill()
tr=d[d.year<=2020]
mo=LogisticRegression(max_iter=5000).fit(tr.dropna(subset=['onset']+fo)[fo],tr.dropna(subset=['onset']+fo).onset)
mw=LogisticRegression(max_iter=5000).fit(tr.dropna(subset=['onset1000']+fw)[fw],tr.dropna(subset=['onset1000']+fw).onset1000)
def score(row):
    if np.expm1(row.ldeaths)>=25 or np.expm1(row.ldeaths_prev)>=25:   # in or near conflict: escalation model
        return mw.predict_proba(row[fw].values.reshape(1,-1))[0,1],'escalation'
    if all(pd.notna(row[f]) for f in fo): return mo.predict_proba(row[fo].values.reshape(1,-1))[0,1],'new onset'
    return np.nan,''
# (a) test wars: rank in year before
cy=pd.read_csv('data/OrganizedViolenceCYDataSet26_1.csv'); p=cy.pivot(index='country',columns='year',values='sb_total_deaths_best').fillna(0)
wars=[(c,y) for c in p.index for y in range(2021,2026) if p.loc[c,y]>=1000 and p.loc[c,y-1]<1000 and p.loc[c,y-2]<1000]
rows=[]
for c,y in wars:
    yr=d[d.year==y-1].copy()
    yr=yr[yr.onset1000.notna()]  # at risk of escalation (below 1000 two years)
    sc=[]
    for _,r in yr.iterrows():
        if all(pd.notna(r[f]) for f in fw): sc.append((r.country,mw.predict_proba(r[fw].values.reshape(1,-1))[0,1]))
    s=pd.DataFrame(sc,columns=['country','p']).sort_values('p',ascending=False).reset_index(drop=True)
    if c in s.country.values:
        rk=int(s.index[s.country==c][0])+1; rows.append((c,y,rk,len(s),float(s[s.country==c].p.iloc[0])))
a=pd.DataFrame(rows,columns=['country','war_year','rank','of','p']).sort_values('rank')
print(a.to_string(index=False))
# (b) 2026 ranking from 2025 data (2025 rows lack 'onset' since 2026 unknown) -> rebuild 2025 rows from panel features
last=d[d.year==2024].copy()   # panel rows are for years <2025 (need t+1); use 2025 deaths directly
# construct 2025 feature rows
p25=p[2025]; p24=p[2024]
feat=[]
v=pd.read_csv('data/V-Dem-CY-Core-v16.csv',usecols=['country_name','year']+inst,low_memory=False); v=v[v.year==2025]
m={'Democratic Republic of the Congo':'DR Congo (Zaire)','Republic of the Congo':'Congo','Russia':'Russia (Soviet Union)','Burma/Myanmar':'Myanmar (Burma)','Yemen':'Yemen (North Yemen)','Cambodia':'Cambodia (Kampuchea)','Türkiye':'Turkey','Bosnia and Herzegovina':'Bosnia-Herzegovina','Serbia':'Serbia (Yugoslavia)','North Macedonia':'Macedonia, FYR','Zimbabwe':'Zimbabwe (Rhodesia)','Eswatini':'Swaziland','Timor-Leste':'East Timor','Madagascar':'Madagascar (Malagasy)','Czechia':'Czech Republic','The Gambia':'Gambia','Palestine/West Bank':'Palestine','Palestine/Gaza':'Palestine'}
v['country']=v.country_name.map(lambda c:m.get(c,c)); v=v.drop_duplicates('country').set_index('country')
mil24=last.set_index('country').mil
import pickle
a24=d[d.year==2024].set_index('country')
for c in p.index:
    cur=p25[c]; prev=p24[c]
    hist_years=[yy for yy in p.columns if yy<=2025 and p.loc[c,yy]>=25]
    # since46/ever46 from ACD via 2024 row then update
    if c not in a24.index: continue
    s46=a24.loc[c,'since46']+1 if cur<25 else 0
    e46=a24.loc[c,'ever46']+(1 if cur>=25 else 0)
    r=dict(country=c,since46=s46,ever46=e46,ldeaths=np.log1p(cur),ldeaths_prev=np.log1p(prev),mil=mil24.get(c,np.nan))
    for f in inst: r[f]=v.loc[c,f] if c in v.index else np.nan
    if cur>=1000: continue  # already at war
    feat.append(r)
F=pd.DataFrame(feat)
out=[]
for _,r in F.iterrows():
    pr,kind=score(r); out.append((r.country,pr,kind,np.expm1(r.ldeaths)))
b=pd.DataFrame(out,columns=['country','p','model','deaths2025']).dropna().sort_values('p',ascending=False)
b.to_csv('panels/rank2026.csv',index=False); a.to_csv('panels/testwars_rank.csv',index=False)
print(b.head(25).to_string(index=False))
