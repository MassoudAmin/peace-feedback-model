"""Can other inputs or other methods raise the forecast of a new conflict? (Chapter 11; Appendix D)

Follows precursor_ranking.py and uses the same panel, outcome, at-risk definition, test windows,
logistic regression, and bootstrap by country. Five tests:

 1. Neighbours. State-based deaths in the countries that share a land border (panels/borders.csv).
 2. Displacement. Internally displaced persons counted by the IOM Displacement Tracking Matrix
    (data/iom_dtm_admin0.csv, country level).
 3. Country size. Population and land area, one value per country (panels/country_size.csv).
 4. Method. Gradient boosting, and the mean of boosting and logistic regression, on the same inputs.
 5. Selection. Forward selection of input groups that uses only the fitting years: within each
    window the last five fitting years serve as the check, and the test years are never seen.

Each new input is scored three ways: added to conflict history, added to conflict history plus
recent state-based deaths (the working model), and in the silent months (no state-based death in
the past year). Every gain carries a 95 percent interval from resampling countries.

Run from the package root, after code/monthly_panel.py:  python code/precursor_pathways.py
"""
import warnings, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); rng = np.random.default_rng(0)

d = pd.read_csv('panels/monthly_panel.csv'); countries = sorted(d.country.unique()); months = sorted(d.ym.unique()); C, T = len(countries), len(months)
ci = {c: i for i, c in enumerate(countries)}; mi = {m: i for i, m in enumerate(months)}
G = lambda col: d.pivot(index='country', columns='ym', values=col).reindex(index=countries, columns=months).values.astype(float)
sb, ns, os_, a_pv, a_fat, a_demo, food = [G(c) for c in ['sb', 'ns', 'os', 'a_pv', 'a_fat', 'a_demo', 'food']]
yearv = np.array([int(m[:4]) for m in months])
p = pd.read_csv('panels/panel2.csv')
def yearly(col):
    w = p.pivot(index='country', columns='year', values=col).reindex(index=countries, columns=range(1990, 2026)).ffill(axis=1)
    return w.reindex(columns=yearv - 1).values.astype(float)
def roll(x, k, mean=False):
    c = np.cumsum(np.nan_to_num(x), 1); out = c.copy(); out[:, k:] = c[:, k:] - c[:, :-k]; out[:, :k - 1] = np.nan
    bad = np.cumsum(np.isnan(x), 1); b = bad.copy(); b[:, k:] = bad[:, k:] - bad[:, :-k]; out[b > 0] = np.nan
    return out / k if mean else out
s12 = roll(sb, 12)
at_risk = np.zeros((C, T), bool); at_risk[:, 23:] = (s12[:, 23:] < 25) & (s12[:, 11:-12] < 25)
silent = at_risk & (s12 == 0)
h = 12; y = np.full((C, T), np.nan)
for t in range(T - h): y[:, t] = (s12[:, t + 1:t + h + 1].max(1) >= 25).astype(float)
y[~at_risk] = np.nan
L = np.log1p; S = lambda *a: np.stack(a, 2)

# ---- neighbours: deaths across the land borders
B = np.zeros((C, C)); bd = pd.read_csv('panels/borders.csv')
for a, b in zip(bd.country, bd.neighbour): B[ci[a], ci[b]] = B[ci[b], ci[a]] = 1
has_nb = B.sum(1) > 0
nb3, nb12 = B @ np.nan_to_num(roll(sb, 3)), B @ np.nan_to_num(s12)
nb_active = B @ (np.nan_to_num(s12) >= 25).astype(float)                     # neighbours at or above 25 deaths in a year
nb_max = np.max(B[:, :, None] * np.nan_to_num(s12)[None, :, :], axis=1)      # the worst neighbour
NEIGH = S(L(nb3), L(nb12), nb_active, L(nb_max)); NEIGH[:, :23] = np.nan
# ---- displacement: IOM DTM, country level, carried forward six months
IOM = {'Antigua and Barbuda': 'Antigua & Barbuda', 'Bahamas (the)': 'Bahamas', 'Bolivia (Plurinational State of)': 'Bolivia', 'Democratic Republic of The Congo': 'DR Congo (Zaire)',
       'Democratic Republic of the Congo': 'DR Congo (Zaire)', "Lao People's Democratic Republic": 'Laos', 'Madagascar': 'Madagascar (Malagasy)', 'Philippines (the)': 'Philippines',
       'Republic of Armenia': 'Armenia', 'Sri lanka': 'Sri Lanka', 'Syrian Arab Republic': 'Syria', 'Yemen': 'Yemen (North Yemen)', 'Zimbabwe': 'Zimbabwe (Rhodesia)'}
idp = np.full((C, T), np.nan)
try:
    q = pd.read_csv('data/iom_dtm_admin0.csv'); q['country'] = q.admin0Name.replace(IOM)
    q['ym'] = q.yearReportingDate.astype(str) + '-' + q.monthReportingDate.astype(str).str.zfill(2)
    for (c, m), v in q.groupby(['country', 'ym']).idp.max().items():
        if c in ci and m in mi: idp[ci[c], mi[m]] = v
    idp = pd.DataFrame(idp).ffill(axis=1, limit=6).values
except FileNotFoundError: print('data/iom_dtm_admin0.csv not found; displacement is skipped')
idp12 = np.full((C, T), np.nan); idp12[:, 12:] = L(idp[:, 12:]) - L(idp[:, :-12])
DISP = S(L(idp), np.nan_to_num(idp12))
# ---- country size
sz = pd.read_csv('panels/country_size.csv').set_index('country').reindex(countries)
SIZE = S(np.repeat(np.log(sz.population.values.astype(float))[:, None], T, 1), np.repeat(np.log(sz.area.values.astype(float))[:, None], T, 1))

GROUPS = {
 'Conflict history since 1946':         S(L(yearly('since46')), L(yearly('ever46'))),
 'State-based deaths, recent months':   S(L(sb), L(roll(sb, 3)), L(s12), L(roll(sb, 24))),
 'Institutional levels':                S(yearly('v2x_rule'), yearly('v2x_polyarchy'), yearly('v2x_civlib'), yearly('v2x_corr'), yearly('v2x_freexp_altinf')),
 'Institutional change, three years':   S(yearly('v2x_rule_d3'), yearly('v2x_polyarchy_d3'), yearly('v2x_corr_d3'), yearly('v2x_freexp_altinf_d3')),
 'Military burden':                     S(yearly('mil'), yearly('mil3')),
 'Other deaths (non-state, one-sided)': S(L(roll(ns + os_, 3)), L(roll(ns + os_, 12))),
 'ACLED violent events and fatalities': S(L(roll(a_pv, 3)), L(roll(a_pv, 12)), L(roll(a_fat, 3)), L(roll(a_fat, 12))),
 'ACLED demonstrations':                S(L(roll(a_demo, 3)), L(roll(a_demo, 12))),
 'Food price inflation':                S(np.clip(food / 100, -1, 5), np.clip(roll(food, 3, True) / 100, -1, 5)),
 'Deaths in neighbouring countries':    NEIGH,
 'Displacement (IOM)':                  DISP,
 'Country size':                        SIZE,
}
NAMES = list(GROUPS); HIST, DEATHS = NAMES[0], NAMES[1]; NEW = NAMES[-3:]
def fill(X):
    miss = np.isnan(X).any(2, keepdims=True).astype(float); return np.concatenate([np.nan_to_num(X), miss], 2)
F = {n: fill(X) for n, X in GROUPS.items()}
cover = {n: float((~np.isnan(GROUPS[n]).any(2))[at_risk].mean()) for n in NAMES}
def design(names): return np.concatenate([F[n] for n in names], 2)
def fit_predict(X, y, tr, method='logit'):
    k = tr & ~np.isnan(y); keep = X[k].std(0) > 0
    if not keep.any(): return np.full((C, T), 0.5)
    Xa = X.reshape(-1, X.shape[2])[:, keep]
    if method == 'boost':
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, l2_regularization=1.0, random_state=0).fit(X[k][:, keep], y[k])
        return m.predict_proba(Xa)[:, 1].reshape(C, T)
    sc = StandardScaler().fit(X[k][:, keep]); m = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X[k][:, keep]), y[k])
    return m.predict_proba(sc.transform(Xa))[:, 1].reshape(C, T)
def auc(pr, rows):
    k = rows & ~np.isnan(y); return roc_auc_score(y[k], pr[k]) if 0 < y[k].sum() < k.sum() else np.nan
WINDOWS = [(2005, 2006, 2010), (2010, 2011, 2015), (2015, 2016, 2020), (2020, 2021, 2025)]
def rows(lo, hi): te = np.zeros((C, T), bool); te[:, (yearv >= lo) & (yearv <= hi)] = True; return te
def train(t_end, t_start=None):
    tr = np.zeros((C, T), bool); t1 = mi[f'{t_end}-12'] + 1; tr[:, 24:t1 - h] = True
    if t_start: tr[:, :mi[f'{t_start}-01']] = False
    return tr & at_risk
TE = [rows(lo, hi) & at_risk for (_, lo, hi) in WINDOWS]; TS = [rows(lo, hi) & silent for (_, lo, hi) in WINDOWS]
P = {}
def run(key, names, method='logit'):
    if key not in P: P[key] = [fit_predict(design(names), y, train(e), method) for (e, _, _) in WINDOWS]
    return key
def A(key, R=TE): return np.array([auc(P[key][w], R[w]) for w in range(4)])
def boot(a_, b_, R=TE, n=300):
    out = []
    for _ in range(n):
        cs = rng.integers(0, C, C); v = []
        for w in range(4):
            r = R[w][cs] & ~np.isnan(y[cs]); yy = y[cs][r]
            if 0 < yy.sum() < len(yy): v.append(roc_auc_score(yy, P[a_][w][cs][r]) - roc_auc_score(yy, P[b_][w][cs][r]))
        out.append(np.mean(v))
    return np.percentile(out, [2.5, 97.5])
def line(label, a_, b_):
    g = A(a_) - A(b_); lo, hi = boot(a_, b_); gs = A(a_, TS) - A(b_, TS); ls, hs = boot(a_, b_, TS)
    print(f'  {label:44s} {g.mean():+.3f} ({lo:+.3f} to {hi:+.3f})  {int((g > 0).sum())}/4    silent months {gs.mean():+.3f} ({ls:+.3f} to {hs:+.3f})  {int((gs > 0).sum())}/4')

n_on = [int(np.nansum(y[TE[w]])) for w in range(4)]
print(f'onset within {h} months; four test windows; mean AUC over windows; at-risk country-months {sum(int(TE[w].sum()) for w in range(4))}, of them silent {sum(int(TS[w].sum()) for w in range(4))}')
run('hist', [HIST]); run('work', [HIST, DEATHS]); run('work+inst', [HIST, DEATHS, 'Institutional levels'])
print(f'conflict history                       all {A("hist").mean():.3f}   silent {A("hist", TS).mean():.3f}')
print(f'history + state-based deaths (working)  all {A("work").mean():.3f}   silent {A("work", TS).mean():.3f}')
print(f'working + institutional levels          all {A("work+inst").mean():.3f}   silent {A("work+inst", TS).mean():.3f}')
print('\ncoverage of at-risk months: ' + '; '.join(f'{n} {cover[n]:.2f}' for n in NEW))
print('\n1-3. new inputs: gain in AUC (95% interval), windows with a gain')
for n in NEW:
    print(n)
    line('added to conflict history', run('hist+' + n, [HIST, n]), 'hist')
    line('added to the working model', run('work+' + n, [HIST, DEATHS, n]), 'work')
    line('added to working + institutions', run('work+inst+' + n, [HIST, DEATHS, 'Institutional levels', n]), 'work+inst')
print('Institutional levels, for reference'); line('added to the working model', 'work+inst', 'work')

print('\n4. method, on the same inputs: gain of the method over logistic regression')
SETS = {'working model': [HIST, DEATHS], 'working + institutions + neighbours': [HIST, DEATHS, 'Institutional levels', 'Deaths in neighbouring countries'], 'all twelve groups': NAMES}
for lab, names in SETS.items():
    a = run('L:' + lab, names); b = run('B:' + lab, names, 'boost'); P['E:' + lab] = [(P[a][w] + P[b][w]) / 2 for w in range(4)]
    print(f'{lab}: logistic {A(a).mean():.3f} (silent {A(a, TS).mean():.3f}); boosting {A(b).mean():.3f} ({A(b, TS).mean():.3f}); mean of the two {A("E:" + lab).mean():.3f} ({A("E:" + lab, TS).mean():.3f})')
    line('boosting minus logistic', b, a); line('mean of the two minus logistic', 'E:' + lab, a)

print('\n5. forward selection inside the fitting years (check = last five fitting years; stop when the gain is under 0.002)')
sel_pred = []
for w, (e, lo, hi) in enumerate(WINDOWS):
    inner_tr = train(e - 5); inner_te = train(e, e - 4); chosen = [HIST]; best = auc(fit_predict(design(chosen), y, inner_tr), inner_te)
    while True:
        cand = [(auc(fit_predict(design(chosen + [n]), y, inner_tr), inner_te), n) for n in NAMES if n not in chosen]
        if not cand: break
        s, n = max(cand)
        if s - best < 0.002: break
        chosen.append(n); best = s
    sel_pred.append(fit_predict(design(chosen), y, train(e)))
    print(f'  window {lo}-{hi}: chosen ' + '; '.join(chosen[1:]) + f'   inner AUC {best:.3f}   test AUC {auc(sel_pred[-1], TE[w]):.3f}')
P['sel'] = sel_pred
print(f'  selected model: all {A("sel").mean():.3f}   silent {A("sel", TS).mean():.3f}')
line('selected model minus the working model', 'sel', 'work')
line('all twelve groups minus the working model', 'L:all twelve groups', 'work')

print('\nwhere the remaining error sits: onsets in the test windows, split by whether the country was silent')
tot = 0
for w in range(4):
    k = TE[w] & (y == 1); ks = TS[w] & (y == 1); tot += 1
    print(f'  window {w + 1}: onset country-months {int(k.sum())}, silent {int(ks.sum())}; AUC working model {auc(P["work"][w], TE[w]):.3f}, silent only {auc(P["work"][w], TS[w]):.3f}, not silent {auc(P["work"][w], TE[w] & ~silent):.3f}')

# ---------------------------------------------------------------- 6. the regression of Table D.9 with and without the neighbours, three horizons
print('\n6. neighbours added to the regression of Table D.9 (monthly_observer.py, model L2: conflict history, rule of law, electoral democracy, UCDP monthly deaths)')
cov = np.stack([G(c) for c in ['since46', 'ever46', 'v2x_rule', 'v2x_polyarchy']], 2); cov[:, :, :2] = np.log1p(cov[:, :, :2])
U = np.nan_to_num(S(L(sb), L(roll(sb, 3)), L(s12), L(roll(sb, 24)), L(roll(ns + os_, 3)), L(roll(ns + os_, 12)))); NB = np.nan_to_num(NEIGH)
for h in [3, 6, 12]:
    y = np.full((C, T), np.nan)
    for t in range(T - h): y[:, t] = (s12[:, t + 1:t + h + 1].max(1) >= 25).astype(float)
    y[~at_risk] = np.nan; P = {'L2': [], 'L2+nb': [], 'L2+nb:boost': []}
    for (e, lo, hi) in WINDOWS:
        t1 = mi[f'{e}-12'] + 1; cm = np.nanmean(cov[:, :t1].reshape(-1, 4), 0); cs_ = np.nanstd(cov[:, :t1].reshape(-1, 4), 0); covz = np.nan_to_num((cov - cm) / cs_)
        P['L2'].append(fit_predict(np.concatenate([covz, U], 2), y, train(e))); P['L2+nb'].append(fit_predict(np.concatenate([covz, U, NB], 2), y, train(e)))
        P['L2+nb:boost'].append(fit_predict(np.concatenate([covz, U, NB], 2), y, train(e), 'boost'))
    P['L2+nb:mean'] = [(P['L2+nb'][w] + P['L2+nb:boost'][w]) / 2 for w in range(4)]
    print(f'horizon {h:2d} months: Table D.9 regression {A("L2").mean():.3f} (silent {A("L2", TS).mean():.3f}); with neighbours {A("L2+nb").mean():.3f} ({A("L2+nb", TS).mean():.3f}), by window ' + ' '.join(f'{x:.3f}' for x in A('L2+nb')) + f'; mean with boosting {A("L2+nb:mean").mean():.3f} ({A("L2+nb:mean", TS).mean():.3f})')
    line('neighbours added', 'L2+nb', 'L2'); line('mean with boosting minus logistic', 'L2+nb:mean', 'L2+nb')
    if h == 12:
        # the alarm list: the top tenth of at-risk country-months in each window, and the onset months inside it
        for key in ['L2', 'L2+nb']:
            hit = tot = n10 = 0
            for w in range(4):
                k = TE[w] & ~np.isnan(y); cut = np.quantile(P[key][w][k], 0.9); top = k & (P[key][w] >= cut)
                hit += int(y[top].sum()); tot += int(y[k].sum()); n10 += int(top.sum())
            print(f'  top tenth, {key}: {n10} country-months hold {hit} of {tot} onset months ({hit / tot:.0%}); one flagged month in {n10 / hit:.1f} is followed by an onset')
