"""loops4.py -- additional loops (Table 3 rows for youth share and contagion; Appendix D, Table D1).

Inputs (source files go in data/; run from the repository root):
  panels/panel2.csv            country-year panel built by fit3.py (onset, onset1000, ldeaths, history, V-Dem, military share)
  API_SP.POP.0014.TO.ZS_*.csv  World Bank, population ages 0-14 (% of total), wide format
  API_SP.POP.GROW_*.csv        World Bank, population growth (annual %)
  API_EN.CLC.SPEI.XD_*.csv     World Bank, SPEI drought index
  GEDEvent_v26_1.csv           UCDP GED 26.1 (contagion: share of land neighbours with >=25 deaths in t-1)
  adjacency.csv                land-border pairs, columns country_a, country_b (UCDP names)
Outputs:
  panels/panel_loops.csv       panel2 plus youth10, popgrow, spei, contagion
  printed: onset rate by youth-share tercile; test AUC of the onset model with each loop added
"""
import glob, pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pycountry

def wb(pattern, name):
    f = glob.glob(pattern)[0]; w = pd.read_csv(f, skiprows=4)
    long = w.melt(id_vars=['Country Name','Country Code'], value_vars=[c for c in w.columns if c[:2] in ('19','20')],
                  var_name='year', value_name=name)
    long['year'] = long.year.astype(int); return long
wbmap = {'RUS':'Russia (Soviet Union)','MMR':'Myanmar (Burma)','VNM':'Vietnam (North Vietnam)','ZWE':'Zimbabwe (Rhodesia)',
         'SRB':'Serbia (Yugoslavia)','GBR':'United Kingdom','USA':'United States of America','CZE':'Czech Republic',
         'KOR':'South Korea','IRN':'Iran','BOL':'Bolivia','VEN':'Venezuela','MDA':'Moldova','TUR':'Turkey','YEM':'Yemen (North Yemen)',
         'COD':'DR Congo (Zaire)','COG':'Congo','CIV':'Ivory Coast','MKD':'Macedonia, FYR','BIH':'Bosnia-Herzegovina',
         'KHM':'Cambodia (Kampuchea)','LAO':'Laos','SYR':'Syria','TZA':'Tanzania','EGY':'Egypt','KGZ':'Kyrgyzstan','SVK':'Slovakia'}
def nm(code):
    if code in wbmap: return wbmap[code]
    x = pycountry.countries.get(alpha_3=code); return x.name if x else None

p = pd.read_csv('panels/panel2.csv')
y = wb('data/API_SP.POP.0014.TO.ZS_*.csv', 'youth'); y['country'] = y['Country Code'].map(nm); y['year'] = y.year + 10  # lag ten years
p = p.merge(y[['country','year','youth']].rename(columns={'youth':'youth10'}), on=['country','year'], how='left')
gr = wb('data/API_SP.POP.GROW_*.csv', 'popgrow'); gr['country'] = gr['Country Code'].map(nm)
p = p.merge(gr[['country','year','popgrow']], on=['country','year'], how='left')
sp = wb('data/API_EN.CLC.SPEI.XD_*.csv', 'spei'); sp['country'] = sp['Country Code'].map(nm)
p = p.merge(sp[['country','year','spei']], on=['country','year'], how='left')

g = pd.read_csv('data/GEDEvent_v26_1.csv', usecols=['year','type_of_violence','country','best'])
g = g[g.type_of_violence == 1].groupby(['country','year']).best.sum().reset_index()
active = set(zip(g[g.best >= 25].country, g[g.best >= 25].year))
adj = pd.read_csv('data/adjacency.csv'); nb = {}
for a, b in zip(adj.country_a, adj.country_b): nb.setdefault(a, set()).add(b); nb.setdefault(b, set()).add(a)
p['contagion'] = [np.mean([(n, yr-1) in active for n in nb[c]]) if c in nb and nb[c] else 0.0 for c, yr in zip(p.country, p.year)]
p.to_csv('panels/panel_loops.csv', index=False)

base = ['since46','ever46','v2x_polyarchy','v2x_rule','v2x_corr','v2x_freexp_altinf','mil']   # columns of panel2.csv
at = p[p.at_risk == True].dropna(subset=['onset'])
def auc(cols):
    x = at.dropna(subset=cols + ['onset']); tr = x[(x.year>=1991)&(x.year<=2020)]; te = x[x.year>=2021]
    m = LogisticRegression(max_iter=3000).fit(tr[cols], tr.onset)
    return round(roc_auc_score(te.onset, m.predict_proba(te[cols])[:,1]), 3), len(te), int(te.onset.sum())
print('base', auc(base))
for extra in [['youth10'], ['contagion'], ['youth10','contagion'], ['popgrow'], ['spei']]:
    print('+', extra, auc(base + extra))
x = at.dropna(subset=['youth10']).copy(); x['q'] = pd.qcut(x.youth10, 3, labels=['old','mid','young'])
print('onset rate by youth-share tercile (%):', (100*x.groupby('q').onset.mean()).round(2).to_dict())
