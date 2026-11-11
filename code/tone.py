import pandas as pd, numpy as np, glob, warnings; warnings.filterwarnings('ignore')
parts=[]
for f in sorted(glob.glob('data/events.20*.tab')):
    d=pd.read_csv(f,sep='\t',usecols=['Event Date','CAMEO Code','Intensity','Country'],low_memory=False,quoting=3,on_bad_lines='skip')
    parts.append(d)
d=pd.concat(parts); d['ym']=pd.to_datetime(d['Event Date'],errors='coerce').dt.to_period('M'); d=d.dropna(subset=['ym','Country'])
d['root']=d['CAMEO Code'].astype(str).str.zfill(3).str[:2].astype(int)
d['hostile']=(d.root>=14).astype(int)          # CAMEO 14-20: protest through unconventional mass violence
d['verbal_conf']=d.root.between(9,13).astype(int) # 09-13: demand, disapprove, reject, threaten
m=d.groupby(['Country','ym']).agg(n=('Intensity','size'),intensity=('Intensity','mean'),hostile=('hostile','mean'),verbal=('verbal_conf','mean')).reset_index()
m=m[m.n>=20]
fx={'Democratic Republic of Congo':'DR Congo (Zaire)','Russian Federation':'Russia (Soviet Union)','Myanmar':'Myanmar (Burma)','Yemen':'Yemen (North Yemen)','Cambodia':'Cambodia (Kampuchea)','Bosnia and Herzegovina':'Bosnia-Herzegovina','Serbia':'Serbia (Yugoslavia)','Zimbabwe':'Zimbabwe (Rhodesia)',"Côte d'Ivoire":'Ivory Coast','Madagascar':'Madagascar (Malagasy)','United States':'United States of America','Occupied Palestinian Territory':'Palestine','Republic of Congo':'Congo','Macedonia':'Macedonia, FYR','Czech Republic':'Czech Republic','Viet Nam':'Vietnam'}
m['country']=m.Country.map(lambda c: fx.get(c,c))
m.to_csv('panels/tone_2018_2019.csv',index=False)
print('country-months',len(m),'countries',m.country.nunique())
# wars/conflicts beginning 2020 or early 2021 after quiet 2018-2019
cy=pd.read_csv('data/OrganizedViolenceCYDataSet26_1.csv')
p=cy.pivot(index='country',columns='year',values='sb_total_deaths_best').fillna(0)
on=[c for c in p.index if p.loc[c,2018]<25 and p.loc[c,2019]<25 and p.loc[c,2020]>=25]
on21=[c for c in p.index if p.loc[c,2019]<25 and p.loc[c,2020]<25 and p.loc[c,2021]>=25]
war=[c for c in p.index if p.loc[c,2018]<1000 and p.loc[c,2019]<1000 and p.loc[c,2020]>=1000]
print('new conflict 2020:',on); print('new conflict 2021:',on21); print('war 2020:',war)
# last 6 months of 2019 vs all 2018-2019 for each country: z-score of hostility within the country's own history
m['half']=m.ym.apply(lambda p: p>=pd.Period('2019-07',freq='M'))
g=m.groupby('country')
base=g.apply(lambda x: x[~x.half][['intensity','hostile','verbal','n']].mean()).rename(columns=lambda c:c+'_base')
late=g.apply(lambda x: x[x.half][['intensity','hostile','verbal','n']].mean()).rename(columns=lambda c:c+'_late')
t=base.join(late).dropna()
t['d_hostile']=t.hostile_late-t.hostile_base; t['d_int']=t.intensity_late-t.intensity_base; t['d_n']=np.log(t.n_late/t.n_base)
t['flag']=['conflict 2020' if c in on else ('war 2020' if c in war else ('conflict 2021' if c in on21 else '')) for c in t.index]
print('\nchange in hostile-event share, H2 2019 vs 2018-H1 2019 (share points):')
print('  all countries median %.3f ; 90th pct %.3f'%(t.d_hostile.median(),t.d_hostile.quantile(.9)))
print(t[t.flag!=''][['hostile_base','hostile_late','d_hostile','d_int','d_n','flag']].round(3).sort_values('d_hostile',ascending=False).to_string())
print('\nrank of each onset country by d_hostile among %d countries (1 = biggest rise):'%len(t))
t['rank']=t.d_hostile.rank(ascending=False).astype(int)
print(t[t.flag!=''][['rank','flag']].sort_values('rank').to_string())
