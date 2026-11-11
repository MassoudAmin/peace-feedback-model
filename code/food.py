import pandas as pd, numpy as np, glob, warnings; warnings.filterwarnings('ignore')
files=sorted(set(glob.glob('data/wfp_food_prices_global_*.csv')))
seen=set(); parts=[]
for f in files:
    y=f.split('global_')[1][:4]
    if y in seen: continue
    seen.add(y)
    d=pd.read_csv(f,usecols=['countryiso3','date','category','commodity','pricetype','price','priceflag'],low_memory=False)
    d=d[(d.category=='cereals and tubers')&(d.pricetype=='Retail')&(d.priceflag=='actual')]
    parts.append(d)
d=pd.concat(parts); d['ym']=pd.to_datetime(d.date).dt.to_period('M')
# country-month staple index: for each country-commodity, median price across markets; log; then average commodities' 12-month log change
cm=d.groupby(['countryiso3','commodity','ym']).price.median().reset_index()
cm['lp']=np.log(cm.price)
cm=cm.sort_values(['countryiso3','commodity','ym'])
cm['chg12']=cm.groupby(['countryiso3','commodity']).lp.diff(12)
# keep only consecutive-month diffs (drop if gap)
cm['gap']=cm.groupby(['countryiso3','commodity']).ym.diff().apply(lambda x: x.n if pd.notna(x) else np.nan)
idx=cm.groupby(['countryiso3','ym']).chg12.median().reset_index()
idx['chg12']=idx.chg12*100
idx.to_csv('panels/staple_chg.csv',index=False)
print('countries',idx.countryiso3.nunique(),'country-months',len(idx),'span',idx.ym.min(),idx.ym.max())
