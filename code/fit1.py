import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# --- UCDP country-year
cy=pd.read_csv('data/OrganizedViolenceCYDataSet26_1.csv')
cy=cy[['country','year','sb_total_deaths_best','ns_total_deaths_best','os_total_deaths_best']]
p=cy.pivot(index='country',columns='year',values='sb_total_deaths_best').fillna(0)
# --- FAO real index annual
fao=pd.read_csv('data/food_price_indices_data.csv',skiprows=2)
fao=fao.iloc[:, :2]; fao.columns=['date','fpi']; fao=fao.dropna(); fao=fao[fao.date.str.match(r'\d{4}-\d{2}')]
fao['year']=fao.date.str[:4].astype(int); fao['fpi']=fao.fpi.astype(float)
fa=fao.groupby('year').fpi.mean(); fchg=fa.pct_change()*100
# --- SIPRI share of GDP
s=pd.read_excel('data/SIPRI-Milex-data-1949-2025_v1_2.xlsx',sheet_name='Share of GDP',header=5)
s=s.rename(columns={s.columns[0]:'country'}).drop(columns=['Notes'],errors='ignore')
s=s[s.country.notna()]
yrs=[c for c in s.columns if isinstance(c,(int,float)) and 1988<=c<=2025]
s=s.set_index('country')[yrs].apply(pd.to_numeric,errors='coerce')*100
s=s[s.notna().sum(axis=1)>5]
# name map SIPRI->UCDP
m={'Congo, DR':'DR Congo (Zaire)','Congo, Republic':'Congo','Côte d’Ivoire':'Ivory Coast',"Cote d'Ivoire":'Ivory Coast','Russia':'Russia (Soviet Union)','Myanmar':'Myanmar (Burma)','Yemen':'Yemen (North Yemen)','Cambodia':'Cambodia (Kampuchea)','Iran':'Iran','Türkiye':'Turkey','Turkiye':'Turkey','Korea, North':'North Korea','Korea, South':'South Korea','Viet Nam':'Vietnam','Bosnia and Herzegovina':'Bosnia-Herzegovina','Serbia':'Serbia (Yugoslavia)','North Macedonia':'Macedonia, FYR','Zimbabwe':'Zimbabwe (Rhodesia)','Eswatini':'Swaziland','Timor-Leste':'East Timor','United States of America':'United States of America','UAE':'United Arab Emirates','Central African Rep.':'Central African Republic','Dominican Rep.':'Dominican Republic','Brunei':'Brunei','Trinidad & Tobago':'Trinidad and Tobago','Madagascar':'Madagascar (Malagasy)','Kyrgyz Republic':'Kyrgyzstan','Kyrgyzstan':'Kyrgyzstan','Czechia':'Czech Republic','Slovakia':'Slovakia','Belarus':'Belarus','Tanzania':'Tanzania','Laos':'Laos','Philippines':'Philippines'}
s.index=[m.get(c,c) for c in s.index]
ucdp_names=set(p.index)
matched=[c for c in s.index if c in ucdp_names]
print('SIPRI countries matched to UCDP:',len(matched),'of',len(s))
# --- panel
rows=[]
years=sorted(p.columns)
for c in p.index:
    for i,y in enumerate(years):
        if y<1991 or y>=2025: continue
        if i<2: continue
        prev=p.loc[c,years[i-1]]; prev2=p.loc[c,years[i-2]]; cur=p.loc[c,y]; nxt=p.loc[c,y+1]
        # onset next year: >=25 deaths in y+1 after <25 in y and y-1
        at_risk = cur<25 and prev<25
        onset = int(nxt>=25) if at_risk else np.nan
        onset1000 = int(nxt>=1000) if (cur<1000 and prev<1000) else np.nan
        hist=(p.loc[c,[yy for yy in years if yy<=y]]>=25).sum()
        last=max([yy for yy in years if yy<=y and p.loc[c,yy]>=25],default=None)
        since= (y-last) if last else 40
        mil=s.loc[c,y] if c in s.index and y in s.columns else np.nan
        mil3=(s.loc[c,y]-s.loc[c,y-3]) if (c in s.index and y-3 in s.columns) else np.nan
        rows.append(dict(country=c,year=y,at_risk=at_risk,onset=onset,onset1000=onset1000,
            ldeaths=np.log1p(cur),ldeaths_prev=np.log1p(prev),since=since,hist=hist,
            fpi=fa.get(y,np.nan),fchg=fchg.get(y,np.nan),mil=mil,mil3=mil3))
d=pd.DataFrame(rows)
d.to_csv('panels/panel.csv',index=False)
def run(target,feats,label):
    dd=d.dropna(subset=[target]+feats)
    tr=dd[dd.year<=2020]; te=dd[dd.year>=2021]
    lr=LogisticRegression(max_iter=2000,C=1.0).fit(tr[feats],tr[target])
    a1=roc_auc_score(tr[target],lr.predict_proba(tr[feats])[:,1]); a2=roc_auc_score(te[target],lr.predict_proba(te[feats])[:,1])
    print(f'{label:34s} n_train={len(tr)} onsets={int(tr[target].sum())} | test n={len(te)} onsets={int(te[target].sum())} | AUC train {a1:.3f} test {a2:.3f}')
    return lr,dd
base=['since','hist']
print('\n== Onset of conflict (>=25 deaths next year, country not in conflict for 2 years)')
run('onset',base,'history only')
run('onset',base+['fpi','fchg'],'+ food price index (global)')
run('onset',base+['mil','mil3'],'+ military share of GDP')
lr,dd=run('onset',base+['fpi','fchg','mil','mil3'],'+ both')
print('coefficients:',dict(zip(base+['fpi','fchg','mil','mil3'],np.round(lr.coef_[0],3))))
print('\n== Escalation to war (>=1000 next year, below 1000 for 2 years)')
run('onset1000',base+['ldeaths','ldeaths_prev'],'history + current deaths')
run('onset1000',base+['ldeaths','ldeaths_prev','fpi','fchg'],'+ food price')
run('onset1000',base+['ldeaths','ldeaths_prev','mil','mil3'],'+ military share')
# thresholds
print('\n== Values in the year before onset vs. other at-risk years (median)')
dd=d[d.at_risk]
for f in ['fpi','fchg','mil','mil3']:
    a=dd[dd.onset==1][f].dropna(); b=dd[dd.onset==0][f].dropna()
    print(f'{f:6s} before onset {a.median():6.2f} (n={len(a)})   no onset {b.median():6.2f} (n={len(b)})')
# revenge / persistence at yearly level
ww=d.dropna(subset=['onset1000'])
print('\nescalation rate by current deaths band:')
ww['band']=pd.cut(np.expm1(ww.ldeaths),[-1,0,24,99,499,999])
print(ww.groupby('band').onset1000.agg(['mean','count']))
