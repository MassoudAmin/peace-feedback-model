"""trust.py -- World Values Survey wave 7 and the trust condition (Tables 5-6, Figures 3-4, Section 6.8).

Inputs (source files go in data/; run from the repository root):
  WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv   WVS wave 7 cross-national file (registration required; not redistributed)
  GEDEvent_v26_1.csv                                 UCDP GED 26.1 (regional test)
  panels/panel2.csv                                  country-year panel built by fit3.py
Outputs:
  panels/wvs7_country_measures_weighted.csv          weighted country means (redistributable)
  panels/wvs7_regions.csv                            regional trust and deaths, 2020-2025
  printed: tercile tables, two-by-two, correlations, regional test
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
import statsmodels.api as sm, pycountry

cols = ['B_COUNTRY_ALPHA','A_YEAR','N_REGION_ISO','O1_LONGITUDE','O2_LATITUDE','W_WEIGHT',
        'Q57P','Q120','Q118','Q112','Q148P','Q146P','Q151P','Q194','Q191','Q131P','Q69P','Q70P','Q71P','Q65P']
d = pd.read_csv('data/WVS_Cross-National_Wave_7_inverted_csv_v6_0.csv', usecols=lambda c: c in cols, low_memory=False)

def share(s, w, cond):
    m = s > 0; return np.average(cond(s[m]).astype(float), weights=w[m])*100 if m.sum() else np.nan
def wavg(s, w):
    m = s > 0; return np.average(s[m], weights=w[m]) if m.sum() else np.nan

rows = []
for c, g in d.groupby('B_COUNTRY_ALPHA'):
    w = g.W_WEIGHT; mf = g.Q151P >= 0
    rows.append(dict(iso=c, year=g.A_YEAR.iloc[0], n=len(g),
        trust=share(g.Q57P, w, lambda s: s == 2),                 # most people can be trusted
        enf_risk=wavg(g.Q120, w),                                 # perceived risk of punishment for a bribe, 1-10
        bribe_freq=wavg(g.Q118, w), corrupt=wavg(g.Q112, w),
        worry_civilwar=share(g.Q148P, w, lambda s: s == 4), worry_war=share(g.Q146P, w, lambda s: s == 4),
        fight=np.average((g.Q151P[mf] == 1), weights=w[mf])*100 if mf.sum() else np.nan,
        polviol_just=wavg(g.Q194, w), viol_just=wavg(g.Q191, w),
        secure=share(g.Q131P, w, lambda s: s == 4),
        conf_state=np.nanmean([share(g[q], w, lambda s: s >= 3) for q in ['Q69P','Q70P','Q71P']]),
        conf_army=share(g.Q65P, w, lambda s: s >= 3)))
c = pd.DataFrame(rows).set_index('iso')
m = {'RUS':'Russia (Soviet Union)','MMR':'Myanmar (Burma)','VNM':'Vietnam (North Vietnam)','ZWE':'Zimbabwe (Rhodesia)',
     'SRB':'Serbia (Yugoslavia)','GBR':'United Kingdom','NIR':None,'USA':'United States of America','CZE':'Czech Republic',
     'KOR':'South Korea','IRN':'Iran','TWN':'Taiwan','HKG':None,'MAC':None,'PRI':None,'BOL':'Bolivia','VEN':'Venezuela',
     'MDA':'Moldova','TUR':'Turkey'}
def nm(i):
    if i in m: return m[i]
    x = pycountry.countries.get(alpha_3=i); return x.name if x else None
c['country'] = [nm(i) for i in c.index]; c = c.dropna(subset=['country'])
c.to_csv('panels/wvs7_country_measures_weighted.csv')

p = pd.read_csv('panels/panel2.csv')
t = p[p.year >= 2020].merge(c, on='country'); t['anyev'] = t[['onset','onset1000']].max(axis=1)
print('countries', t.country.nunique(), 'rows', len(t))
def terc(col, out):
    x = t.dropna(subset=[out, col]).copy(); x['g'] = pd.qcut(x[col], 3, labels=['low','mid','high'])
    g = x.groupby('g')[out].agg(['size','mean']); g['mean'] = (100*g['mean']).round(1); return g
print('onset by trust tercile\n', terc('trust','onset')); print('escalation by trust tercile\n', terc('trust','onset1000'))
def two(e, tr, label):
    x = t.dropna(subset=['anyev', e, tr]).copy()
    x['E'] = np.where(x[e] >= x[e].median(), 'high', 'low'); x['T'] = np.where(x[tr] >= x[tr].median(), 'high', 'low')
    g = x.groupby(['E','T']).anyev.agg(['size','mean']); g['mean'] = (100*g['mean']).round(1); print(label); print(g.unstack())
two('v2x_rule', 'trust', 'V-Dem rule of law x trust (Table 6)')
two('enf_risk', 'trust', 'survey enforcement (Q120) x trust')
cy = t.groupby('country').agg(trust=('trust','first'), enf=('enf_risk','first'), pv=('polviol_just','first'),
                              wcw=('worry_civilwar','first'), cs=('conf_state','first'), ld=('ldeaths','max'), rule=('v2x_rule','mean'))
for k in ['trust','enf','pv','wcw','cs']:
    r, pv = spearmanr(cy[k], cy.ld, nan_policy='omit'); print(f'{k}: rho={r:.2f} p={pv:.3f} n={cy[k].notna().sum()}')
r, pv = spearmanr(cy.enf, cy.rule, nan_policy='omit'); print('survey enforcement vs V-Dem rule of law: rho', round(r,2))

# ---- regional test: regional trust vs state-based deaths 2020-2025 assigned to nearest regional centroid
dr = d[(d.O1_LONGITUDE > -900) & (d.O2_LATITUDE > -900) & (d.N_REGION_ISO.astype(str).str.len() > 2)]
ged = pd.read_csv('data/GEDEvent_v26_1.csv', usecols=['year','type_of_violence','country','latitude','longitude','best'])
ged = ged[(ged.type_of_violence == 1) & (ged.year >= 2020)]
out = []
for iso, name in c.country.items():
    s = dr[dr.B_COUNTRY_ALPHA == iso]
    if len(s) == 0: continue
    reg = s.groupby('N_REGION_ISO').apply(lambda x: pd.Series(dict(lat=x.O2_LATITUDE.median(), lon=x.O1_LONGITUDE.median(), n=len(x),
        trust=np.average((x.Q57P[x.Q57P>0]==2), weights=x.W_WEIGHT[x.Q57P>0])*100 if (x.Q57P>0).sum() >= 30 else np.nan))).dropna()
    if len(reg) < 4: continue
    e = ged[ged.country == name]; deaths = np.zeros(len(reg))
    if len(e):
        la = np.radians(reg.lat.values)[:,None]; lo = np.radians(reg.lon.values)[:,None]
        ela = np.radians(e.latitude.values)[None,:]; elo = np.radians(e.longitude.values)[None,:]
        dist = np.arccos(np.clip(np.sin(la)*np.sin(ela)+np.cos(la)*np.cos(ela)*np.cos(lo-elo), -1, 1))
        deaths = np.bincount(dist.argmin(axis=0), weights=e.best.values, minlength=len(reg))
    reg['deaths'] = deaths; reg['country'] = name; out.append(reg.reset_index())
R = pd.concat(out); R['ld'] = np.log1p(R.deaths); R.to_csv('panels/wvs7_regions.csv', index=False)
S = R[R.groupby('country').deaths.transform('sum') >= 100].copy()
S['t_dm'] = S.trust - S.groupby('country').trust.transform('mean'); S['l_dm'] = S.ld - S.groupby('country').ld.transform('mean')
mfit = sm.WLS(S.l_dm, sm.add_constant(S.t_dm), weights=np.sqrt(S.n)).fit(cov_type='cluster', cov_kwds={'groups': S.country})
print('within-country slope of log deaths on trust:', round(mfit.params.t_dm,3), 'p', round(mfit.pvalues.t_dm,3),
      'regions', len(S), 'countries', S.country.nunique())
