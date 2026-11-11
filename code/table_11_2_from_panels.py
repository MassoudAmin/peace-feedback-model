"""table_11_2_from_panels.py
Reproduces, from panels/panel2.csv alone (no source downloads):
  - Table 11.2 of The Coupled World (the posted fits: counts and AUCs)
  - Table D.2 (the same models refitted without penalty on standardized inputs)
  - the held-out checks reported in Chapter 11 and Appendix D: base rate, Brier score,
    calibration by thirds, top-tenth capture, country-clustered bootstrap, permutation test,
    and the persistence baseline for escalation.
Run from the repository root:  python3 code/table_11_2_from_panels.py
Rows are country-years. Predictors in a row use that year or earlier; the outcome is the
following year. Fit: rows 1991-2020. Test: rows 2021-2024 (onsets of 2022-2025).
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
d = pd.read_csv('panels/panel2.csv')
inst = ['v2x_polyarchy', 'v2x_rule', 'v2x_corr', 'v2x_freexp_altinf']; dinst = [c + '_d3' for c in inst]
MODELS = [('Conflict history from 1989', ['since', 'hist']),
          ('Conflict history from 1946', ['since46', 'ever46']),
          ('+ institutional level', ['since46', 'ever46'] + inst),
          ('+ institutional 3-year change', ['since46', 'ever46'] + inst + dinst),
          ('+ military share of GDP', ['since46', 'ever46'] + inst + dinst + ['mil', 'mil3'])]
def split(target, feats):
    dd = d.dropna(subset=[target] + feats); return dd[dd.year <= 2020], dd[dd.year >= 2021]
print('== Table 11.2 (posted fit: scikit-learn default L2 penalty, raw inputs)')
for label, feats in MODELS:
    tr, te = split('onset', feats); lr = LogisticRegression(max_iter=3000).fit(tr[feats], tr.onset)
    print(f'{label:32s} train {len(tr)} / {int(tr.onset.sum())}  test {len(te)} / {int(te.onset.sum())}  AUC train {roc_auc_score(tr.onset, lr.predict_proba(tr[feats])[:,1]):.3f} test {roc_auc_score(te.onset, lr.predict_proba(te[feats])[:,1]):.3f}')
print('\n== Table D.2 (no penalty, standardized inputs; log-odds per standard deviation)')
for label, feats in MODELS:
    tr, te = split('onset', feats); sc = StandardScaler().fit(tr[feats])
    try: lr = LogisticRegression(penalty=None, max_iter=20000).fit(sc.transform(tr[feats]), tr.onset)
    except Exception: lr = LogisticRegression(C=1e12, max_iter=20000).fit(sc.transform(tr[feats]), tr.onset)
    print(f'{label:32s} AUC train {roc_auc_score(tr.onset, lr.predict_proba(sc.transform(tr[feats]))[:,1]):.3f} test {roc_auc_score(te.onset, lr.predict_proba(sc.transform(te[feats]))[:,1]):.3f}')
    print('    ' + '  '.join(f'{f}={c:+.2f}' for f, c in zip(feats, lr.coef_[0])))
print('\n== Held-out checks')
def checks(target, feats, label, boot=True):
    tr, te = split(target, feats); lr = LogisticRegression(max_iter=3000).fit(tr[feats], tr[target])
    p = lr.predict_proba(te[feats])[:,1]; y = te[target].values; pb = tr[target].mean()
    print(f'{label}: test {len(te)} / {int(y.sum())}; base rate {y.mean():.4f}; mean predicted {p.mean():.4f}; AUC {roc_auc_score(y,p):.3f}; Brier {brier_score_loss(y,p):.4f} vs base-rate forecast {np.mean((y-pb)**2):.4f}')
    q = pd.qcut(p, 3, labels=False, duplicates='drop')
    print('    thirds of predicted risk (predicted, observed, n, events):', [(round(float(p[q==k].mean()),4), round(float(y[q==k].mean()),4), int((q==k).sum()), int(y[q==k].sum())) for k in sorted(set(q))])
    k = int(round(0.1*len(y))); print('    events in top tenth of predicted risk:', int(y[np.argsort(-p)[:k]].sum()), 'of', int(y.sum()))
    if boot:
        rng = np.random.default_rng(0); cs = te.country.unique(); idx = {c: np.where(te.country.values == c)[0] for c in cs}; a = []
        for _ in range(2000):
            s = np.concatenate([idx[c] for c in rng.choice(cs, len(cs))])
            if 0 < y[s].sum() < len(s): a.append(roc_auc_score(y[s], p[s]))
        print('    test AUC, 2,000 resamples by country, 2.5th-97.5th percentile: %.2f-%.2f' % tuple(np.percentile(a, [2.5, 97.5])))
    return y, p
checks('onset', MODELS[4][1], 'Fullest new-conflict model')
y, p = checks('onset', MODELS[1][1], 'Conflict history from 1946')
rng = np.random.default_rng(1); obs = roc_auc_score(y, p)
print('    permutation test: shuffles reaching the observed AUC: %d of 5000' % sum(roc_auc_score(rng.permutation(y), p) >= obs for _ in range(5000)))
checks('onset1000', ['ldeaths'], 'Escalation to war, last-year deaths only')
checks('onset1000', ['since46', 'ever46', 'ldeaths', 'ldeaths_prev'], 'Escalation to war, history and deaths')
