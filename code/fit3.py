import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
d=pd.read_csv('panels/panel.csv')
# V-Dem
v=pd.read_csv('data/V-Dem-CY-Core-v16.csv',usecols=['country_name','year','v2x_polyarchy','v2x_rule','v2x_corr','v2x_freexp_altinf','v2x_civlib'],low_memory=False)
m={'Democratic Republic of the Congo':'DR Congo (Zaire)','Republic of the Congo':'Congo','Russia':'Russia (Soviet Union)','Burma/Myanmar':'Myanmar (Burma)','Yemen':'Yemen (North Yemen)','Cambodia':'Cambodia (Kampuchea)','Türkiye':'Turkey','Bosnia and Herzegovina':'Bosnia-Herzegovina','Serbia':'Serbia (Yugoslavia)','North Macedonia':'Macedonia, FYR','Zimbabwe':'Zimbabwe (Rhodesia)','Eswatini':'Swaziland','Timor-Leste':'East Timor','United States of America':'United States of America','Ivory Coast':'Ivory Coast','Madagascar':'Madagascar (Malagasy)','Czechia':'Czech Republic','The Gambia':'Gambia','Vietnam':'Vietnam','Palestine/West Bank':'Palestine','Palestine/Gaza':'Palestine'}
v['country']=v.country_name.map(lambda c:m.get(c,c))
v=v[v.year>=1985].drop(columns='country_name')
# 3-year change in institutions
v=v.sort_values(['country','year'])
for c in ['v2x_polyarchy','v2x_rule','v2x_corr','v2x_freexp_altinf']:
    v[c+'_d3']=v.groupby('country')[c].diff(3)
d=d.merge(v,on=['country','year'],how='left')
print('rows with V-Dem:',d.v2x_rule.notna().mean().round(2))
# ACD history since 1946
a=pd.read_csv('data/UcdpPrioConflict_v26_1.csv')
a['location']=a.location.astype(str)
hist={}
for _,r in a.iterrows():
    for loc in r.location.split(', '):
        hist.setdefault(loc,set()).add(r.year)
def since46(c,y):
    ys=[yy for yy in hist.get(c,[]) if yy<=y]
    return (y-max(ys)) if ys else (y-1946)
d['since46']=[since46(c,y) for c,y in zip(d.country,d.year)]
d['ever46']=[len([yy for yy in hist.get(c,[]) if yy<=y]) for c,y in zip(d.country,d.year)]
def run(target,feats,label):
    dd=d.dropna(subset=[target]+feats)
    tr=dd[dd.year<=2020]; te=dd[dd.year>=2021]
    lr=LogisticRegression(max_iter=3000).fit(tr[feats],tr[target])
    a1=roc_auc_score(tr[target],lr.predict_proba(tr[feats])[:,1]); a2=roc_auc_score(te[target],lr.predict_proba(te[feats])[:,1])
    print(f'{label:44s} train n={len(tr)} on={int(tr[target].sum())} | test n={len(te)} on={int(te[target].sum())} | AUC {a1:.3f} / {a2:.3f}')
    return lr
inst=['v2x_polyarchy','v2x_rule','v2x_corr','v2x_freexp_altinf']
dinst=[c+'_d3' for c in inst]
print('== New conflict onset (>=25 next year after 2 quiet years)')
run('onset',['since','hist'],'history 1989+')
run('onset',['since46','ever46'],'history 1946+')
run('onset',['since46','ever46']+inst,'+ institutions (level)')
run('onset',['since46','ever46']+inst+dinst,'+ institutions (level + 3-yr change)')
lr=run('onset',['since46','ever46']+inst+dinst+['mil','mil3'],'+ military share')
print('coef:',dict(zip(['since46','ever46']+inst+dinst+['mil','mil3'],np.round(lr.coef_[0],2))))
print('\n== Escalation to war (>=1000 next year)')
run('onset1000',['since46','ever46','ldeaths','ldeaths_prev'],'history + deaths')
run('onset1000',['since46','ever46','ldeaths','ldeaths_prev']+inst,'+ institutions')
run('onset1000',['since46','ever46','ldeaths','ldeaths_prev']+inst+dinst,'+ institutions change')
print('\n== Medians in year before onset vs other at-risk years')
dd=d[d.at_risk]
for f in inst+dinst:
    a_=dd[dd.onset==1][f].dropna(); b_=dd[dd.onset==0][f].dropna()
    print(f'{f:24s} onset {a_.median():7.3f}  none {b_.median():7.3f}')
# the "enforcement x trust" claim: does rule of law reduce escalation more where corruption is low?
w=d.dropna(subset=['onset1000','v2x_rule','v2x_corr'])
w=w[np.expm1(w.ldeaths)>=25]
for lab,mk in [('rule of law high, corruption low',(w.v2x_rule>0.5)&(w.v2x_corr<0.5)),('rule of law high, corruption high',(w.v2x_rule>0.5)&(w.v2x_corr>=0.5)),('rule of law low, corruption low',(w.v2x_rule<=0.5)&(w.v2x_corr<0.5)),('rule of law low, corruption high',(w.v2x_rule<=0.5)&(w.v2x_corr>=0.5))]:
    s=w[mk]; print(f'{lab:36s} escalation rate {s.onset1000.mean():.3f}  n={len(s)}')
d.to_csv('panels/panel2.csv',index=False)
