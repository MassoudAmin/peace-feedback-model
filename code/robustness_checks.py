"""robustness_checks.py
Four checks on Table 11.2 of The Coupled World, from panels/panel2.csv alone.
Run from the repository root:  python3 code/robustness_checks.py
  A. the five models scored on one common test sample (rows with every input present)
  B. an unpenalized refit in a second library (statsmodels), for comparison with Table D.2
  C. the test AUC with each held-out onset removed in turn
  D. the test AUC of the history model when the fitting period ends in 2018, 2019, 2020, or 2021
Rows are country-years; predictors use that year or earlier; the outcome is the following year.
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
d = pd.read_csv('panels/panel2.csv')
inst = ['v2x_polyarchy', 'v2x_rule', 'v2x_corr', 'v2x_freexp_altinf']; dinst = [c + '_d3' for c in inst]
MODELS = [('Conflict history from 1989', ['since', 'hist']),
          ('Conflict history from 1946', ['since46', 'ever46']),
          ('+ institutional level', ['since46', 'ever46'] + inst),
          ('+ institutional 3-year change', ['since46', 'ever46'] + inst + dinst),
          ('+ military share of GDP', ['since46', 'ever46'] + inst + dinst + ['mil', 'mil3'])]
def fit_score(tr, te, feats):
    lr = LogisticRegression(max_iter=3000).fit(tr[feats], tr.onset)
    return roc_auc_score(te.onset, lr.predict_proba(te[feats])[:, 1]), lr
print('== A. Common test sample (every input present)')
allf = sorted(set(sum([f for _, f in MODELS], [])))
dd = d.dropna(subset=['onset'] + allf); tr, te = dd[dd.year <= 2020], dd[dd.year >= 2021]
print(f'train {len(tr)} / {int(tr.onset.sum())}  test {len(te)} / {int(te.onset.sum())}')
for label, feats in MODELS:
    print(f'{label:32s} test AUC {fit_score(tr, te, feats)[0]:.3f}')
print('\n== B. Unpenalized refit in statsmodels, standardized inputs')
for label, feats in MODELS[1:3]:
    q = d.dropna(subset=['onset'] + feats); tr, te = q[q.year <= 2020], q[q.year >= 2021]
    mu, sd = tr[feats].mean(), tr[feats].std()
    r = sm.Logit(tr.onset.astype(float), sm.add_constant((tr[feats] - mu) / sd)).fit(disp=0)
    p = r.predict(sm.add_constant((te[feats] - mu) / sd, has_constant='add'))
    print(f'{label:32s} test AUC {roc_auc_score(te.onset, p):.3f}')
print('\n== C. Each held-out onset removed in turn')
for label, feats in (MODELS[1], MODELS[4]):
    q = d.dropna(subset=['onset'] + feats); tr, te = q[q.year <= 2020], q[q.year >= 2021]
    auc, lr = fit_score(tr, te, feats); p = lr.predict_proba(te[feats])[:, 1]; y = te.onset.values
    a = [roc_auc_score(np.delete(y, i), np.delete(p, i)) for i in np.where(y == 1)[0]]
    print(f'{label:32s} test AUC {auc:.3f}; with one onset removed {min(a):.3f} to {max(a):.3f}')
print('\n== D. Fitting period ended in other years (conflict history from 1946)')
feats = MODELS[1][1]; q = d.dropna(subset=['onset'] + feats)
for cut in (2018, 2019, 2020, 2021):
    tr, te = q[q.year <= cut], q[q.year > cut]
    print(f'fit through {cut}: test {len(te)} / {int(te.onset.sum())}  test AUC {fit_score(tr, te, feats)[0]:.3f}')
