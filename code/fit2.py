import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
g=pd.read_csv('data/GEDEvent_v26_1.csv',usecols=['country','date_start','type_of_violence','best'])
g=g[g.type_of_violence==1]
g['ym']=pd.to_datetime(g.date_start).dt.to_period('M')
cm=g.groupby(['country','ym']).best.sum()
idx=pd.period_range('1989-01','2025-12',freq='M')
cm=cm.unstack(fill_value=0).reindex(columns=idx,fill_value=0)
X=np.log1p(cm.values)
# pooled AR(1): x_t = a + b x_{t-1}
x0=X[:,:-1].ravel(); x1=X[:,1:].ravel()
b=np.polyfit(x0,x1,1)[0]
print(f'pooled monthly persistence (log deaths): b = {b:.3f}  (12-month decay of a shock: {b**12:.3f})')
# AR(1) restricted to months with violence but below war level (<100 deaths)
mask=(np.expm1(x0)>0)&(np.expm1(x0)<100)
print(f'same, months with 1-99 deaths: b = {np.polyfit(x0[mask],x1[mask],1)[0]:.3f}')
# probability next month has any deaths, by this month's deaths
d0=np.expm1(x0); d1=np.expm1(x1)
for lo,hi in [(0,0),(1,9),(10,49),(50,199),(200,1e9)]:
    m=(d0>=lo)&(d0<=hi); print(f'deaths this month {lo:>4}-{hi if hi<1e9 else "+":<4}: P(any deaths next month)={np.mean(d1[m]>0):.2f}  P(>=25 next month)={np.mean(d1[m]>=25):.3f}  n={m.sum()}')
# war onset month: first month of a 12-month window reaching 1000 after 24 quiet months (<25/month)
onsets=[]
for i,c in enumerate(cm.index):
    v=cm.iloc[i].values
    for t in range(24,len(v)-12):
        if v[t-24:t].max()<25 and v[t:t+12].sum()>=1000 and v[t]>=25:
            onsets.append((c,idx[t])); break
print('\nmonthly war onsets (1000 deaths within 12 months after 24 quiet months):',len(onsets))
# revenge ramp: median deaths in months -6..-1 relative to onset
ramp=[]
for c,t in onsets:
    j=idx.get_loc(t); ramp.append(cm.loc[c].values[j-6:j+1])
ramp=np.array(ramp)
print('median deaths by month before onset (-6..0):',np.median(ramp,axis=0).astype(int))
print('share of onsets with zero deaths in all of months -6..-1:',np.mean(ramp[:,:-1].sum(axis=1)==0).round(2))
# food price monthly: 12-month change in FPI at onset month vs all months
fao=pd.read_csv('data/food_price_indices_data.csv',skiprows=2).iloc[:,:2]; fao.columns=['date','fpi']
fao=fao[fao.date.astype(str).str.match(r'\d{4}-\d{2}')]; fao['ym']=pd.PeriodIndex(fao.date,freq='M'); fao['fpi']=fao.fpi.astype(float)
f=fao.set_index('ym').fpi.reindex(idx); chg=f.pct_change(12)*100
at=[chg.get(t,np.nan) for c,t in onsets]; at=np.array([a for a in at if not np.isnan(a)])
print(f'\nFAO index 12-month change at war onset month: median {np.median(at):.1f}%  vs all months {chg.median():.1f}%; share of onsets with change>+10%: {np.mean(at>10):.2f} vs base {np.mean(chg.dropna()>10):.2f}')
