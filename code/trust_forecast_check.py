"""Held-out check: does interpersonal trust improve the yearly onset forecast?

Reads only panels/panel2.csv and panels/wvs7_country_measures_weighted.csv.
Fits on country-years 1991-2020 and tests on 2021-2024 (onsets of 2022-2025),
for the 62 countries surveyed in World Values Survey wave 7. Trust is the
wave-7 country mean, applied to every year as a slow variable.

Run from the package root:  python code/trust_forecast_check.py
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
p = pd.read_csv("panels/panel2.csv")
w = pd.read_csv("panels/wvs7_country_measures_weighted.csv")[["country", "trust"]]
d = p.merge(w, on="country")
d = d[d.at_risk == 1].dropna(subset=["onset", "since46", "ever46", "v2x_rule", "trust"])
d["rule_x_trust"] = d.v2x_rule * d.trust
tr = d[(d.year >= 1991) & (d.year <= 2020)]
te = d[(d.year >= 2021) & (d.year <= 2024)].reset_index(drop=True)
print(f"countries {d.country.nunique()}; train {len(tr)} / {int(tr.onset.sum())}; "
      f"test {len(te)} / {int(te.onset.sum())}")
print("test onsets:", ", ".join(te[te.onset == 1].country))

sets = {
    "history": ["since46", "ever46"],
    "history + rule of law": ["since46", "ever46", "v2x_rule"],
    "history + rule of law + trust": ["since46", "ever46", "v2x_rule", "trust"],
    "history + rule of law + trust + product": ["since46", "ever46", "v2x_rule", "trust", "rule_x_trust"],
}
pred = {}
for k, c in sets.items():
    sc = StandardScaler().fit(tr[c])
    m = LogisticRegression(C=1e6, max_iter=5000).fit(sc.transform(tr[c]), tr.onset)
    pred[k] = m.predict_proba(sc.transform(te[c]))[:, 1]
    print(f"{k:42s} train AUC {roc_auc_score(tr.onset, m.predict_proba(sc.transform(tr[c]))[:, 1]):.3f}"
          f"  test AUC {roc_auc_score(te.onset, pred[k]):.3f}"
          f"  Brier {brier_score_loss(te.onset, pred[k]):.4f}")
print(f"base-rate Brier {brier_score_loss(te.onset, np.full(len(te), tr.onset.mean())):.4f}")

rng = np.random.default_rng(0)
cs = te.country.unique()
idx = {c: np.where(te.country == c)[0] for c in cs}
diffs = []
for _ in range(2000):
    s = np.concatenate([idx[c] for c in rng.choice(cs, len(cs))])
    y = te.onset.values[s]
    if 0 < y.sum() < len(y):
        diffs.append(roc_auc_score(y, pred["history + rule of law + trust"][s])
                     - roc_auc_score(y, pred["history + rule of law"][s]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"change in test AUC when trust is added: median {np.median(diffs):.3f}, "
      f"2.5th-97.5th percentile {lo:.3f} to {hi:.3f} ({len(diffs)} resamples by country)")
