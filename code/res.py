import pandas as pd, numpy as np, pycountry, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
r=pd.read_csv('data/API_FI.RES.TOTL.MO_DS2_en_csv_v2_407033.csv',skiprows=4)
fx={'COD':'DR Congo (Zaire)','COG':'Congo','RUS':'Russia (Soviet Union)','MMR':'Myanmar (Burma)','YEM':'Yemen (North Yemen)','KHM':'Cambodia (Kampuchea)','TUR':'Turkey','BIH':'Bosnia-Herzegovina','SRB':'Serbia (Yugoslavia)','MKD':'Macedonia, FYR','ZWE':'Zimbabwe (Rhodesia)','SWZ':'Swaziland','TLS':'East Timor','CIV':'Ivory Coast','MDG':'Madagascar (Malagasy)','IRN':'Iran','SYR':'Syria','LAO':'Laos','BOL':'Bolivia','TZA':'Tanzania','VEN':'Venezuela','PSE':'Palestine','KOR':'South Korea','PRK':'North Korea','VNM':'Vietnam','MDA':'Moldova','GMB':'Gambia','KGZ':'Kyrgyzstan','CZE':'Czech Republic','USA':'United States of America','GBR':'United Kingdom','EGY':'Egypt'}
def nm(i):
    if i in fx: return fx[i]
    c=pycountry.countries.get(alpha_3=i); return c.name if c else None
rows=[]
for _,x in r.iterrows():
    c=nm(x['Country Code']); 
    if not c: continue
    for y in range(1988,2026):
        v=x.get(str(y)); 
        if pd.notna(v): rows.append((c,y,float(v)))
rs=pd.DataFrame(rows,columns=['country','year','res_mo'])
d=pd.read_csv('panels/panel3.csv').merge(rs,on=['country','year'],how='left')
d['res_mo']=d.res_mo.clip(0,24); d['lres']=np.log1p(d.res_mo)
print('rows with reserves:',d.res_mo.notna().sum(),'of',len(d))
inst=['v2x_polyarchy','v2x_rule','v2x_corr','v2x_freexp_altinf']
def run(target,feats,label):
    dd=d.dropna(subset=[target]+feats); tr=dd[dd.year<=2020]; te=dd[dd.year>=2021]
    lr=LogisticRegression(max_iter=5000).fit(tr[feats],tr[target])
    print(f'{label:40s} train n={len(tr)} on={int(tr[target].sum())} | test n={len(te)} on={int(te[target].sum())} | AUC {roc_auc_score(tr[target],lr.predict_proba(tr[feats])[:,1]):.3f} / {roc_auc_score(te[target],lr.predict_proba(te[feats])[:,1]):.3f}')
    return lr
for target in ['onset','onset1000']:
    x=d.dropna(subset=[target,'res_mo']); print(f'{target}: reserves (months of imports) in year before, median: onset {x[x[target]==1].res_mo.median():.1f}  none {x[x[target]==0].res_mo.median():.1f}  (n onset={int(x[target].sum())})')
dd=d.dropna(subset=['res_mo']); d=dd
run('onset',['since46','ever46']+inst,'onset: history + institutions')
lr=run('onset',['since46','ever46']+inst+['lres'],'onset: + reserves'); print('  reserves coef',round(lr.coef_[0][-1],3))
run('onset1000',['since46','ever46','ldeaths','ldeaths_prev'],'war: history + deaths')
lr=run('onset1000',['since46','ever46','ldeaths','ldeaths_prev','lres'],'war: + reserves'); print('  reserves coef',round(lr.coef_[0][-1],3))
# the buffer claim: does the food-price effect on escalation depend on reserves?
w=d.dropna(subset=['onset1000','fcpi_max','res_mo']); w=w[np.expm1(w.ldeaths)>=25]
for lab,mk in [('food shock >10%, reserves <3 mo',(w.fcpi_max>10)&(w.res_mo<3)),('food shock >10%, reserves >=3 mo',(w.fcpi_max>10)&(w.res_mo>=3)),('no shock, reserves <3 mo',(w.fcpi_max<=10)&(w.res_mo<3)),('no shock, reserves >=3 mo',(w.fcpi_max<=10)&(w.res_mo>=3))]:
    s=w[mk]; print(f'{lab:36s} escalation rate {s.onset1000.mean():.3f}  n={len(s)}')
d.to_csv('panels/panel4.csv',index=False)
