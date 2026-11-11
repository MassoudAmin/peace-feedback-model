"""Violence inside the country: other kinds of violence, and provinces (Chapter 11; Appendix D)

Part A. Kinds of violence, by country. UCDP records three kinds of organized violence: state-based
(a government is one side), non-state (two organized groups, neither a government: communal,
sectarian, and militia fighting), and one-sided (an organized group or a government kills civilians;
attacks on civilians by organized armed groups fall here). The same forecast is run for the onset of
each kind: trailing-twelve-month deaths of that kind reach 25 within twelve months, in a country
below 25 now and twelve months earlier. Inputs enter in steps: conflict history and institutions;
recent deaths of the same kind; recent deaths of the other two kinds; deaths in neighbouring
countries.

Part B. Provinces. The unit is the first-level administrative area recorded by UCDP for each event
(adm_1). The outcome is the same rule applied to a province, all three kinds of violence summed.
Events located only to a country (where_prec 5 or more) or without a province are left out. In each
window the provinces are those with at least one recorded event by the end of the fitting years, so
the list of units uses nothing from the test years; onsets in provinces with no earlier event are
counted and reported as missed. Inputs enter in steps: what the country forecast knows (country
covariates and country deaths); the province's own record; deaths in the rest of the country and
within 300 km in other provinces.

Logistic regression, four test windows, and bootstrap by country as in precursor_ranking.py.
Run from the package root, after code/monthly_panel.py:  python code/within_country.py
"""
import warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); rng = np.random.default_rng(0)

d = pd.read_csv('panels/monthly_panel.csv'); countries = sorted(d.country.unique()); months = sorted(d.ym.unique()); C, T = len(countries), len(months)
ci = {c: i for i, c in enumerate(countries)}; mi = {m: i for i, m in enumerate(months)}; yearv = np.array([int(m[:4]) for m in months])
G = lambda col: d.pivot(index='country', columns='ym', values=col).reindex(index=countries, columns=months).values.astype(float)
sb, ns, os_ = G('sb'), G('ns'), G('os')
cov = np.stack([G(c) for c in ['since46', 'ever46', 'v2x_rule', 'v2x_polyarchy']], 2); cov[:, :, :2] = np.log1p(cov[:, :, :2])
L = np.log1p
def roll(x, k):
    c = np.cumsum(x, 1); out = c.copy(); out[:, k:] = c[:, k:] - c[:, :-k]; return out
B = np.zeros((C, C)); bd = pd.read_csv('panels/borders.csv')
for a, b in zip(bd.country, bd.neighbour): B[ci[a], ci[b]] = B[ci[b], ci[a]] = 1
WINDOWS = [(2005, 2006, 2010), (2010, 2011, 2015), (2015, 2016, 2020), (2020, 2021, 2025)]; h = 12
def outcome(x):                                   # x: units by months, deaths per month
    s12 = roll(x, 12); n = x.shape[0]; risk = np.zeros((n, T), bool); risk[:, 23:] = (s12[:, 23:] < 25) & (s12[:, 11:-12] < 25)
    y = np.full((n, T), np.nan)
    for t in range(T - h): y[:, t] = (s12[:, t + 1:t + h + 1].max(1) >= 25).astype(float)
    y[~risk] = np.nan; return y, risk, s12
def rec(x): return np.stack([L(x), L(roll(x, 3)), L(roll(x, 12)), L(roll(x, 24))], 2)
def fit_predict(X, y, tr):
    k = tr & ~np.isnan(y); keep = X[k].std(0) > 0; sc = StandardScaler().fit(X[k][:, keep])
    m = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X[k][:, keep]), y[k])
    return m.predict_proba(sc.transform(X.reshape(-1, X.shape[2])[:, keep]))[:, 1].reshape(X.shape[:2])
def auc(p, y, r):
    k = r & ~np.isnan(y); return roc_auc_score(y[k], p[k]) if 0 < y[k].sum() < k.sum() else np.nan
def covz(e):
    t1 = mi[f'{e}-12'] + 1; cm = np.nanmean(cov[:, :t1].reshape(-1, 4), 0); cs = np.nanstd(cov[:, :t1].reshape(-1, 4), 0); return np.nan_to_num((cov - cm) / cs)
def tr_rows(n, e): tr = np.zeros((n, T), bool); tr[:, 24:mi[f'{e}-12'] + 1 - h] = True; return tr
def te_rows(n, lo, hi): te = np.zeros((n, T), bool); te[:, (yearv >= lo) & (yearv <= hi)] = True; return te
def boot(Pa, Pb, y, R, grp, n=300):               # gain a minus b, resampling countries; grp maps each unit to its country
    G_ = int(grp.max()) + 1; members = [np.where(grp == g)[0] for g in range(G_)]; out = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, G_, G_)]); v = []
        for w in range(4):
            r = R[w][idx] & ~np.isnan(y[idx]); yy = y[idx][r]
            if 0 < yy.sum() < len(yy): v.append(roc_auc_score(yy, Pa[w][idx][r]) - roc_auc_score(yy, Pb[w][idx][r]))
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])

# ---------------------------------------------------------------- A. kinds of violence, by country
print('A. Onset of each kind of violence within twelve months, by country; mean AUC over four test windows')
KIND = {'state-based': sb, 'non-state (communal, sectarian, militia)': ns, 'one-sided (killing of civilians)': os_}
nb = np.stack([L(B @ roll(sb + ns + os_, 12)), B @ (roll(sb + ns + os_, 12) >= 25).astype(float)], 2)
grpC = np.arange(C)
for kind, x in KIND.items():
    y, risk, s12 = outcome(x); others = [v for k2, v in KIND.items() if k2 != kind]
    steps = {'conflict history and institutions': lambda cz: cz,
             '+ recent deaths of the same kind': lambda cz: np.concatenate([cz, rec(x)], 2),
             '+ recent deaths of the other two kinds': lambda cz: np.concatenate([cz, rec(x), rec(others[0]), rec(others[1])], 2),
             '+ deaths in neighbouring countries': lambda cz: np.concatenate([cz, rec(x), rec(others[0]), rec(others[1]), nb], 2)}
    P = {s: [] for s in steps}; R = []; RS = []; n_ep = 0
    for (e, lo, hi) in WINDOWS:
        cz = covz(e); tr = tr_rows(C, e) & risk; te = te_rows(C, lo, hi) & risk; R.append(te); RS.append(te & (s12 == 0))
        ep = np.zeros((C, T), bool); ep[:, 1:] = risk[:, :-1] & (s12[:, 1:] >= 25); n_ep += int((ep & te_rows(C, lo, hi)).sum())
        for s, f in steps.items(): P[s].append(fit_predict(f(cz), y, tr))
    A = lambda s, RR=R: np.array([auc(P[s][w], y, RR[w]) for w in range(4)])
    print(f'\n{kind}: {n_ep} onset episodes in the test windows')
    prev = None
    for s in steps:
        a = A(s); txt = f'  {s:42s} {a.mean():.3f} (windows {a.min():.2f}-{a.max():.2f})   silent months {np.nanmean(A(s, RS)):.3f}'
        if prev:
            g = a - A(prev); lo_, hi_ = boot(P[s], P[prev], y, R, grpC); txt += f'   step gain {g.mean():+.3f} ({lo_:+.3f} to {hi_:+.3f}), {int((g > 0).sum())}/4'
        print(txt); prev = s

# ---------------------------------------------------------------- B. provinces
print('\n\nB. Onset of organized violence in a province within twelve months (all three kinds summed)')
u = ['country', 'date_start', 'adm_1', 'latitude', 'longitude', 'best', 'where_prec']
ev = pd.concat([pd.read_csv(f, usecols=u, low_memory=False) for f in ['data/GEDEvent_v26_1.csv', 'data/GEDEvent_v26_01_26_09.csv']])
n_all, d_all = len(ev), ev.best.sum()
ev = ev[ev.adm_1.notna() & (ev.where_prec <= 4) & ev.country.isin(ci)]; ev['ym'] = ev.date_start.str[:7]; ev = ev[ev.ym.isin(mi)]
print(f'events placed in a province: {len(ev)} of {n_all} ({len(ev) / n_all:.0%}); deaths {ev.best.sum() / d_all:.0%} of all recorded deaths')
ev['unit'] = ev.country + ' | ' + ev.adm_1; units = sorted(ev.unit.unique()); ui = {k: i for i, k in enumerate(units)}; N = len(units)
X = np.zeros((N, T)); np.add.at(X, (ev.unit.map(ui).values, ev.ym.map(mi).values), ev.best.values)
first = ev.groupby('unit').ym.min().reindex(units).map(mi).values                      # month of the first recorded event
cen = ev.groupby('unit')[['latitude', 'longitude']].mean().reindex(units).values; uc = np.array([ci[k.split(' | ')[0]] for k in units])
la, lo_ = np.radians(cen[:, 0]), np.radians(cen[:, 1])
dist = 6371 * np.arccos(np.clip(np.sin(la)[:, None] * np.sin(la)[None, :] + np.cos(la)[:, None] * np.cos(la)[None, :] * np.cos(lo_[:, None] - lo_[None, :]), -1, 1))
near = ((dist <= 300) & ~np.eye(N, dtype=bool)).astype(float)
Ctot = np.zeros((C, T)); np.add.at(Ctot, uc, X)
y, risk, s12 = outcome(X); x12 = roll(X, 12)
since = np.zeros((N, T)); last = np.full(N, -1.0)                                      # months since the province was last at 25 deaths in a year
for t in range(T):
    last = np.where(s12[:, t] >= 25, t, last); since[:, t] = np.where(last >= 0, t - last, t + 120)
own = np.concatenate([rec(X), L(since)[:, :, None], L(np.cumsum(X, 1))[:, :, None]], 2)
rest = np.stack([L(roll(Ctot, 3)[uc] - roll(X, 3)), L(roll(Ctot, 12)[uc] - x12), L(near @ roll(X, 3)), L(near @ x12), near @ (x12 >= 25).astype(float)], 2)
ctry = np.stack([L(roll(Ctot, 3))[uc], L(roll(Ctot, 12))[uc], L(roll(Ctot, 24))[uc]], 2)
steps = {'what the country forecast knows': lambda cz: np.concatenate([cz[uc], ctry], 2),
         "+ the province's own record": lambda cz: np.concatenate([cz[uc], ctry, own], 2),
         '+ rest of the country and provinces within 300 km': lambda cz: np.concatenate([cz[uc], ctry, own, rest], 2),
         "the province's own record alone": lambda cz: own}
P = {s: [] for s in steps}; R = []; RS = []
print(f'provinces with a recorded event, 1989 to September 2026: {N}, in {len(set(uc))} countries')
for (e, lo, hi) in WINDOWS:
    t1 = mi[f'{e}-12'] + 1; known = first < t1; cz = covz(e)
    tr = tr_rows(N, e) & risk & known[:, None] & (np.arange(T)[None, :] >= first[:, None])
    te = te_rows(N, lo, hi) & risk & known[:, None]; R.append(te); RS.append(te & (s12 == 0))
    ep = np.zeros((N, T), bool); ep[:, 1:] = risk[:, :-1] & (s12[:, 1:] >= 25); ep &= te_rows(N, lo, hi)
    print(f'  window {lo}-{hi}: provinces known {int(known.sum())}; onset episodes {int(ep.sum())}, of them in provinces with no earlier event {int(ep[~known].sum())}; at-risk province-months {int(te.sum())}, onset rate {np.nanmean(y[te]):.3f}')
    for s, f in steps.items(): P[s].append(fit_predict(f(cz), y, tr))
A = lambda s, RR=R: np.array([auc(P[s][w], y, RR[w]) for w in range(4)])
prev = None
for s in steps:
    a = A(s); txt = f'  {s:52s} {a.mean():.3f} (windows {a.min():.2f}-{a.max():.2f})   silent province-months {np.nanmean(A(s, RS)):.3f}'
    if prev and not s.endswith('alone'):
        g = a - A(prev); l_, h_ = boot(P[s], P[prev], y, R, uc); txt += f'   step gain {g.mean():+.3f} ({l_:+.3f} to {h_:+.3f}), {int((g > 0).sum())}/4'
    print(txt); prev = s
for s in list(steps)[:3]:                          # the alarm list
    hit = tot = n10 = 0
    for w in range(4):
        k = R[w] & ~np.isnan(y); cut = np.quantile(P[s][w][k], 0.9); top = k & (P[s][w] >= cut); hit += int(y[top].sum()); tot += int(y[k].sum()); n10 += int(top.sum())
    print(f'  top tenth, {s}: holds {hit / tot:.0%} of onset province-months; one flagged month in {n10 / hit:.1f} is followed by an onset')
# inside countries only: is the right province picked among the provinces of one country? (AUC within country-month groups, pooled by demeaning)
print('\nwithin a country: mean AUC over countries with at least one onset and one quiet province-month in the window')
for s in list(steps)[:3]:
    v = []
    for w in range(4):
        for g in np.unique(uc):
            m_ = (uc == g); r = R[w][m_] & ~np.isnan(y[m_]); yy = y[m_][r]
            if 20 <= yy.sum() < len(yy) - 20: v.append(roc_auc_score(yy, P[s][w][m_][r]))
    print(f'  {s:52s} {np.mean(v):.3f}   ({len(v)} country-windows)')
