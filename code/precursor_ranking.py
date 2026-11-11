"""Which precursors carry the warning, and does the system change? (Chapter 11; Appendix D)

Three questions, on the country-month panel of monthly_panel.py with the yearly covariates of
panels/panel2.csv and the monthly news tone of panels/tone_all.csv (ICEWS, 1995-2022):

 1. Identification. Ten groups of precursors are ranked by what each adds, out of sample, to the
    forecast of a new conflict within twelve months: alone, added to conflict history, and removed
    from the full model. The same is done for the silent months (no state-based death in a year).
 2. Uncertainty. The gain of each group over conflict history is resampled by country (bootstrap),
    with the four test windows kept separate, to give a 95 percent interval.
 3. Adaptation. Does the relation between precursors and onset change over time? Three ways of
    fitting are compared on the same test windows: a model fitted once on the years to 2005 and
    never updated, a model refitted on all earlier years before each window, and a model refitted
    on only the ten years before each window.

Outcome and at-risk definition as in monthly_observer.py. Logistic regression throughout, with the
same L2 penalty, so that differences come from the inputs. A missing input is set to zero with an
indicator. Four windows: fit through 2005 / 2010 / 2015 / 2020; test 2006-2010 / 2011-2015 /
2016-2020 / 2021-September 2025.

Run from the package root, after code/monthly_panel.py:  python code/precursor_ranking.py
"""
import warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); rng = np.random.default_rng(0)

d = pd.read_csv('panels/monthly_panel.csv'); countries = sorted(d.country.unique()); months = sorted(d.ym.unique()); C, T = len(countries), len(months)
G = lambda col: d.pivot(index='country', columns='ym', values=col).reindex(index=countries, columns=months).values.astype(float)
sb, ns, os_, a_pv, a_fat, a_demo, food = [G(c) for c in ['sb', 'ns', 'os', 'a_pv', 'a_fat', 'a_demo', 'food']]
yearv = np.array([int(m[:4]) for m in months]); mi = {m: i for i, m in enumerate(months)}
p = pd.read_csv('panels/panel2.csv')
def yearly(col):                                   # yearly covariate, lagged one year, carried forward
    w = p.pivot(index='country', columns='year', values=col).reindex(index=countries, columns=range(1990, 2026)).ffill(axis=1)
    return w.reindex(columns=yearv - 1).values.astype(float)
tone = pd.read_csv('panels/tone_all.csv')
def tonem(col):
    return tone.pivot_table(index='country', columns='ym', values=col).reindex(index=countries, columns=months).values.astype(float)
hostile, intensity, nnews = tonem('hostile'), tonem('intensity'), tonem('n')

def roll(x, k, mean=False):
    c = np.cumsum(np.nan_to_num(x), 1); out = c.copy(); out[:, k:] = c[:, k:] - c[:, :-k]; out[:, :k - 1] = np.nan
    bad = np.cumsum(np.isnan(x), 1); b = bad.copy(); b[:, k:] = bad[:, k:] - bad[:, :-k]; out[b > 0] = np.nan
    return out / k if mean else out
s12 = roll(sb, 12)
at_risk = np.zeros((C, T), bool); at_risk[:, 23:] = (s12[:, 23:] < 25) & (s12[:, 11:-12] < 25)
silent = at_risk & (s12 == 0)
def target(h):
    y = np.full((C, T), np.nan)
    for t in range(T - h): y[:, t] = (s12[:, t + 1:t + h + 1].max(1) >= 25).astype(float)
    y[~at_risk] = np.nan; return y
L = np.log1p; S = lambda *a: np.stack(a, 2)
GROUPS = {
 'Conflict history since 1946':        S(L(yearly('since46')), L(yearly('ever46'))),
 'Institutional levels':               S(yearly('v2x_rule'), yearly('v2x_polyarchy'), yearly('v2x_civlib'), yearly('v2x_corr'), yearly('v2x_freexp_altinf')),
 'Institutional change, three years':  S(yearly('v2x_rule_d3'), yearly('v2x_polyarchy_d3'), yearly('v2x_corr_d3'), yearly('v2x_freexp_altinf_d3')),
 'Military burden':                    S(yearly('mil'), yearly('mil3')),
 'State-based deaths, recent months':  S(L(sb), L(roll(sb, 3)), L(s12), L(roll(sb, 24))),
 'Other deaths (non-state, one-sided)': S(L(roll(ns + os_, 3)), L(roll(ns + os_, 12))),
 'ACLED violent events and fatalities': S(L(roll(a_pv, 3)), L(roll(a_pv, 12)), L(roll(a_fat, 3)), L(roll(a_fat, 12))),
 'ACLED demonstrations':               S(L(roll(a_demo, 3)), L(roll(a_demo, 12))),
 'Food price inflation':               S(np.clip(food / 100, -1, 5), np.clip(roll(food, 3, True) / 100, -1, 5)),
 'News tone (ICEWS)':                  S(roll(hostile, 3, True), roll(hostile, 12, True), roll(intensity, 3, True), L(roll(nnews, 3))),
}
NAMES = list(GROUPS); HIST = NAMES[0]
def fill(X):
    miss = np.isnan(X).any(2, keepdims=True).astype(float); return np.concatenate([np.nan_to_num(X), miss], 2)
F = {n: fill(X) for n, X in GROUPS.items()}
cover = {n: float((~np.isnan(GROUPS[n]).any(2))[at_risk].mean()) for n in NAMES}
def design(names): return np.concatenate([F[n] for n in names], 2)
def fit_predict(X, y, tr):
    k = tr & ~np.isnan(y); keep = X[k].std(0) > 0
    if not keep.any(): return np.full((C, T), 0.5)          # the input has no record yet in the fitting years
    sc = StandardScaler().fit(X[k][:, keep])
    m = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X[k][:, keep]), y[k])
    return m.predict_proba(sc.transform(X.reshape(-1, X.shape[2])[:, keep]))[:, 1].reshape(C, T)
def auc(pr, y, rows):
    k = rows & ~np.isnan(y); return roc_auc_score(y[k], pr[k]) if 0 < y[k].sum() < k.sum() else np.nan
WINDOWS = [(2005, 2006, 2010), (2010, 2011, 2015), (2015, 2016, 2020), (2020, 2021, 2025)]
h = 12; y = target(h)
def rows(lo, hi): te = np.zeros((C, T), bool); te[:, (yearv >= lo) & (yearv <= hi)] = True; return te
def train(t_end, start_year=None):
    tr = np.zeros((C, T), bool); t1 = mi[f'{t_end}-12'] + 1; t0 = 24 if start_year is None else max(24, mi[f'{start_year}-01'])
    tr[:, t0:t1 - h] = True; return tr & at_risk

# ---------------------------------------------------------------- 1. ranking
SPECS = {'hist': [HIST], 'full': NAMES}
for n in NAMES:
    SPECS['alone:' + n] = [n]
    if n != HIST: SPECS['hist+' + n] = [HIST, n]
    SPECS['full-' + n] = [m for m in NAMES if m != n]
P = {s: [] for s in SPECS}
for (tr_end, lo, hi) in WINDOWS:
    tr = train(tr_end)
    for s, names in SPECS.items(): P[s].append(fit_predict(design(names), y, tr))
TE = [rows(lo, hi) & at_risk for (_, lo, hi) in WINDOWS]; TS = [rows(lo, hi) & silent for (_, lo, hi) in WINDOWS]
def A(s, R=TE): return np.array([auc(P[s][w], y, R[w]) for w in range(4)])
def boot_gain(a_, b_, R=TE, n=300):
    out = []
    for _ in range(n):
        cs = rng.integers(0, C, C); v = []
        for w in range(4):
            r = R[w][cs] & ~np.isnan(y[cs]); yy = y[cs][r]
            if 0 < yy.sum() < len(yy): v.append(roc_auc_score(yy, P[a_][w][cs][r]) - roc_auc_score(yy, P[b_][w][cs][r]))
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])
print(f'onset within {h} months; four test windows; mean AUC over windows')
print(f'conflict history alone {A("hist").mean():.3f}   all ten groups {A("full").mean():.3f}   silent months: {A("hist", TS).mean():.3f} and {A("full", TS).mean():.3f}\n')
tab = []
for n in NAMES:
    alone = A('alone:' + n).mean(); drop = (A('full') - A('full-' + n)).mean()
    if n == HIST: gain, lo_, hi_, pos, gs = np.nan, np.nan, np.nan, 0, np.nan
    else:
        g = A('hist+' + n) - A('hist'); gain = g.mean(); pos = int((g > 0).sum()); lo_, hi_ = boot_gain('hist+' + n, 'hist'); gs = (A('hist+' + n, TS) - A('hist', TS)).mean()
    tab.append((n, cover[n], alone, gain, lo_, hi_, pos, drop, gs))
tab.sort(key=lambda r: -(r[3] if r[3] == r[3] else 9))
print(f'{"precursor group":38s} cover  alone  gain over history (95% interval)  windows>0  loss if removed  gain, silent months')
for n, cv, al, g, lo_, hi_, pos, dr, gs in tab:
    if n == HIST: print(f'{n:38s} {cv:5.2f}  {al:.3f}  (the base)                                     {dr:+.3f}')
    else: print(f'{n:38s} {cv:5.2f}  {al:.3f}  {g:+.3f}  ({lo_:+.3f} to {hi_:+.3f})          {pos}/4        {dr:+.3f}          {gs:+.3f}')

# ---------------------------------------------------------------- 3. adaptation
print('\nadaptation: AUC by test window for three ways of fitting')
for label, names in [('conflict history + state-based deaths', [HIST, 'State-based deaths, recent months']), ('all ten groups', NAMES)]:
    X = design(names); res = {'fitted once, to 2005': [], 'refitted on all earlier years': [], 'refitted on the last ten years': []}
    frozen = fit_predict(X, y, train(2005))
    for w, (tr_end, lo, hi) in enumerate(WINDOWS):
        res['fitted once, to 2005'].append(auc(frozen, y, TE[w]))
        res['refitted on all earlier years'].append(auc(fit_predict(X, y, train(tr_end)), y, TE[w]))
        res['refitted on the last ten years'].append(auc(fit_predict(X, y, train(tr_end, tr_end - 9)), y, TE[w]))
    print('  ' + label)
    for k, v in res.items(): print(f'    {k:32s}' + '  '.join(f'{x:.3f}' for x in v) + f'   mean of windows 2-4: {np.mean(v[1:]):.3f}')
# standardized weights by window, small model (one institutional index, to keep the weights readable)
print('\nstandardized logistic weights by fitting window (conflict history, rule of law, state-based deaths):')
X = np.nan_to_num(np.concatenate([GROUPS[HIST], GROUPS['Institutional levels'][:, :, :1], GROUPS['State-based deaths, recent months']], 2))
labels = ['log years since conflict', 'log conflict years since 1946', 'rule of law', 'deaths, 1 month', 'deaths, 3 months', 'deaths, 12 months', 'deaths, 24 months']
W = []
for (tr_end, lo, hi) in WINDOWS:
    k = train(tr_end) & ~np.isnan(y); sc = StandardScaler().fit(X[k]); W.append(LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X[k]), y[k]).coef_[0])
for j, lab in enumerate(labels): print(f'    {lab:32s}' + '  '.join(f'{W[w][j]:+.2f}' for w in range(4)))
