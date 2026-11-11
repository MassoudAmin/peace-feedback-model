"""trust_pair_from_panels.py
Reproduces, from the shipped panels alone (no source downloads), the yearly results of
trust.py and pair.py that the book reports in Chapter 11:
  - new-conflict onset and escalation by tercile of interpersonal trust (Table 11.3)
  - the rule-of-law by trust two-by-two (Table 11.4) and the trust-deaths rank correlation (Figure 11.3)
  - the pair-level escalation ladder, the pair escalation AUC, and the reciprocity coefficients
Reads panels/panel2.csv, panels/wvs7_country_measures_weighted.csv, panels/pair_year_panel.csv.
Run from the repository root:  python3 code/trust_pair_from_panels.py
The monthly pair ladder, the interstate subset, and the regional trust test need the source
files and are produced by pair.py and trust.py.
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
print('== Trust (WVS wave 7 country means joined to the country-year panel, rows from 2020)')
c = pd.read_csv('panels/wvs7_country_measures_weighted.csv'); p = pd.read_csv('panels/panel2.csv')
t = p[p.year >= 2020].merge(c.drop(columns=['year']), on='country'); t['anyev'] = t[['onset', 'onset1000']].max(axis=1)
print('countries', t.country.nunique(), 'rows', len(t))
for out in ['onset', 'onset1000']:
    x = t.dropna(subset=[out, 'trust']).copy(); x['g'] = pd.qcut(x.trust, 3, labels=['low', 'mid', 'high'])
    g = x.groupby('g')[out].agg(['size', 'sum']); g['percent'] = (100*g['sum']/g['size']).round(1); print(out, 'by trust tercile'); print(g)
x = t.dropna(subset=['anyev', 'v2x_rule', 'trust']).copy()
x['rule'] = np.where(x.v2x_rule >= x.v2x_rule.median(), 'high', 'low'); x['trust_'] = np.where(x.trust >= x.trust.median(), 'high', 'low')
g = x.groupby(['rule', 'trust_']).anyev.agg(['size', 'sum']); g['percent'] = (100*g['sum']/g['size']).round(1)
print('onset or escalation by rule of law (V-Dem) and trust'); print(g)
cy = t.groupby('country').agg(trust=('trust', 'first'), ld=('ldeaths', 'max'))
r, pv = spearmanr(cy.trust, cy.ld, nan_policy='omit'); print(f'trust vs peak log deaths: Spearman rho {r:.2f}, p {pv:.3f}, n {cy.trust.notna().sum()}')
print('\n== Pairs (pair-year panel)')
q = pd.read_csv('panels/pair_year_panel.csv'); print('pairs', q.dyad_new_id.nunique(), 'pair-years', len(q))
act = q[(q.best > 0) & (q.best < 1000)].copy()
act['bin'] = pd.cut(act.best, [0, 24, 99, 499, 999], labels=['1-24', '25-99', '100-499', '500-999']); act['war_next'] = (act.best_next >= 1000).astype(int)
for nm, sub in [('fit 1991-2020', act[(act.year >= 1991) & (act.year <= 2020)]), ('test 2021-2024', act[act.year >= 2021])]:
    l = sub.groupby('bin').war_next.agg(['size', 'mean']); l['mean'] = (100*l['mean']).round(1); print(nm); print(l)
act['ld'] = np.log1p(act.best); act['ldp'] = np.log1p(act.best_prev)
tr = act[(act.year >= 1991) & (act.year <= 2020)]; te = act[act.year >= 2021]
m = LogisticRegression(max_iter=2000).fit(tr[['ld', 'ldp']], tr.war_next)
print('pair escalation AUC, test rows 2021-2024:', round(roc_auc_score(te.war_next, m.predict_proba(te[['ld', 'ldp']])[:, 1]), 3), 'n', len(te), 'escalations', int(te.war_next.sum()))
z = q[(q.best > 0) & (q.year >= 1991) & (q.year <= 2024)].copy()
for k in ['da', 'db', 'da_next', 'db_next']: z['l'+k] = np.log1p(z[k])
def ols(y, X):
    A = np.column_stack([np.ones(len(X)), X.values]); return np.linalg.lstsq(A, y.values, rcond=None)[0]
b = ols(z.ldb_next, z[['lda', 'ldb']]); print('non-state side losses next year on state losses this year: %.3f (own losses %.3f), n %d' % (b[1], b[2], len(z)))
b = ols(z.lda_next, z[['ldb', 'lda']]); print('state side losses next year on non-state losses this year: %.3f (own losses %.3f)' % (b[1], b[2]))
