"""pair.py -- pair-level tests (Tables 1 and 2 of the paper).

Inputs (public; place them in data/ and run from the repository root):
  GEDEvent_v26_1.csv      UCDP Georeferenced Event Dataset 26.1
  Dyadic_v26_1.csv        UCDP Dyadic Dataset 26.1
Outputs:
  panels/pair_year_panel.csv    pair-year deaths by side, 1989-2025
  printed: pair escalation ladder, pair escalation AUC, reciprocity regressions,
           monthly pair ladder
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

g = pd.read_csv('data/GEDEvent_v26_1.csv', usecols=['year','type_of_violence','dyad_new_id','dyad_name',
                                                'date_start','deaths_a','deaths_b','best'])
g = g[g.type_of_violence == 1]                      # state-based violence only
g['month'] = pd.to_datetime(g.date_start).dt.to_period('M')

# ---- pair-year panel, every pair filled 1989-2025
dy = g.groupby(['dyad_new_id','year']).agg(da=('deaths_a','sum'), db=('deaths_b','sum'), best=('best','sum'))
names = g.groupby('dyad_new_id').dyad_name.first()
idx = pd.MultiIndex.from_product([dy.index.get_level_values(0).unique(), range(1989, 2026)], names=['dyad_new_id','year'])
p = dy.reindex(idx, fill_value=0).reset_index(); p['dyad_name'] = p.dyad_new_id.map(names)
p = p.sort_values(['dyad_new_id','year'])
for c in ['da','db','best']:
    p[c+'_prev'] = p.groupby('dyad_new_id')[c].shift(1); p[c+'_next'] = p.groupby('dyad_new_id')[c].shift(-1)
p = p.dropna(subset=['da_prev','best_next'])
p.to_csv('panels/pair_year_panel.csv', index=False)
print('pairs', p.dyad_new_id.nunique(), 'pair-years', len(p))

# ---- yearly escalation ladder at pair level (pairs with events and <1,000 deaths)
act = p[(p.best > 0) & (p.best < 1000)].copy()
act['bin'] = pd.cut(act.best, [0,24,99,499,999], labels=['1-24','25-99','100-499','500-999'])
act['war_next'] = (act.best_next >= 1000).astype(int)
for nm, sub in [('fit 1991-2020', act[(act.year>=1991)&(act.year<=2020)]), ('test 2021-2024', act[act.year>=2021])]:
    l = sub.groupby('bin').war_next.agg(['size','mean']); l['mean'] = (100*l['mean']).round(1); print(nm); print(l)
act['ld'] = np.log1p(act.best); act['ldp'] = np.log1p(act.best_prev)
tr = act[(act.year>=1991)&(act.year<=2020)]; te = act[act.year>=2021]
m = LogisticRegression(max_iter=2000).fit(tr[['ld','ldp']], tr.war_next)
print('pair escalation AUC 2021-24:', round(roc_auc_score(te.war_next, m.predict_proba(te[['ld','ldp']])[:,1]),3),
      'n', len(te), 'escalations', int(te.war_next.sum()))

# ---- reciprocity (Table 2): losses of one side in t -> losses of the other in t+1
q = p[(p.best > 0) & (p.year >= 1991) & (p.year <= 2024)].copy()
for c in ['da','db','da_next','db_next']: q['l'+c] = np.log1p(q[c])
def fit(y, X, label, groups):
    r = sm.OLS(y, sm.add_constant(X)).fit(cov_type='cluster', cov_kwds={'groups': groups})
    print(label); print(' coef', r.params.round(3).to_dict()); print(' p', r.pvalues.round(4).to_dict(), 'n', int(r.nobs), 'R2', round(r.rsquared,3))
fit(q.ldb_next, q[['lda','ldb']], 'non-state side losses (t+1) ~ state losses (t) + own (t)', q.dyad_new_id)
fit(q.lda_next, q[['ldb','lda']], 'state side losses (t+1) ~ non-state losses (t) + own (t)', q.dyad_new_id)
dd = pd.read_csv('data/Dyadic_v26_1.csv'); inter = set(dd[dd.type_of_conflict == 2].dyad_id)
qi = q[q.dyad_new_id.isin(inter)]; print('interstate pair-years', len(qi), 'pairs', qi.dyad_new_id.nunique())
fit(qi.ldb_next, qi[['lda','ldb']], 'interstate: side B (t+1)', qi.dyad_new_id)
fit(qi.lda_next, qi[['ldb','lda']], 'interstate: side A (t+1)', qi.dyad_new_id)

# ---- monthly pair ladder (Table 1, pair columns)
mo = g.groupby(['dyad_new_id','month']).best.sum().reset_index()
rows = []
for i, s in mo.groupby('dyad_new_id'):
    s = s.set_index('month').best; ix = pd.period_range(s.index.min(), s.index.max(), freq='M'); s = s.reindex(ix, fill_value=0)
    rows.append(pd.DataFrame({'dyad': i, 'd': s.values, 'dn': np.append(s.values[1:], np.nan)}))
mm = pd.concat(rows).dropna()
mm['bin'] = pd.cut(mm.d, [-1,0,9,49,199,1e9], labels=['0','1-9','10-49','50-199','200+'])
print(mm.groupby('bin').agg(n=('dn','size'), p25=('dn', lambda s: round((s>=25).mean(),3))))
