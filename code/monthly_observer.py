"""A monthly observer with several sensors (Chapter 15, open problem 11; Appendix D).

Question. The yearly estimators of state_estimators.py add one to two points of AUC to a regression
on the same inputs. Does a monthly estimator fed by several sensors do better, and does it recover
anything in the months when the death record shows nothing?

Panel. panels/monthly_panel.csv (built by code/monthly_panel.py): 199 countries, January 1989 to
September 2026, with UCDP deaths by type, ACLED events, fatalities and demonstrations, World Bank
food price inflation, and yearly covariates lagged one year.

Outcome. s12[t] is the sum of state-based deaths over months t-11..t. A country-month is at risk
when s12[t] < 25 and s12[t-12] < 25. The outcome at horizon h is that s12 reaches 25 in some month
t+1..t+h (h = 3, 6, 12). An onset episode is the first such month after an at-risk month.

State-space model, one latent state per country and month:
    z[t+1] = mu + phi (z[t] - mu) + w,   mu = a0 + a . covariates,   w ~ N(0, q)
    y_k[t] = max(0, c_k + l_k z[t] + v_k),  v_k ~ N(0, r_k)     count sensors, log(1 + count)
    y_f[t] = c_f + l_f z[t] + v_f                               food inflation, not censored
The state-based death sensor has c = 0 and l = 1. A zero is a censored reading. The filter is a
Tobit Kalman filter that carries the exact mean and variance of the state given a censored reading;
sensors are processed one at a time and a missing sensor is skipped.

Models, all with conflict history and institutional levels:
    L0  logistic regression, conflict history
    L1  + institutional levels
    L2  + UCDP monthly deaths (state-based over 1, 3, 12 and 24 months; other types over 3 and 12)
    F1  Tobit Kalman filter on state-based deaths, one sensor
    F2  Tobit Kalman filter on six sensors (deaths by two types, ACLED events, fatalities and
        demonstrations, food inflation)
    F3  Tobit Kalman filter on the three sensors that count deaths (UCDP state-based, UCDP other
        types, ACLED fatalities)
    L3  logistic regression on every sensor feature, with indicators for missing sensors
    GB  gradient boosting on every sensor feature
Four rolling windows: fit through 2005 / 2010 / 2015 / 2020, test 2006-2010 / 2011-2015 /
2016-2020 / 2021-September 2025 (outcomes through September 2026).

Run from the package root: python code/monthly_observer.py     (about twenty minutes)
"""
import warnings, sys, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); rng = np.random.default_rng(0)

d = pd.read_csv('panels/monthly_panel.csv')
countries = sorted(d.country.unique()); months = sorted(d.ym.unique()); C, T = len(countries), len(months)
G = lambda col: d.pivot(index='country', columns='ym', values=col).reindex(index=countries, columns=months).values.astype(float)
sb, ns, os_, a_pv, a_fat, a_demo, food = [G(c) for c in ['sb', 'ns', 'os', 'a_pv', 'a_fat', 'a_demo', 'food']]
cov = np.stack([G(c) for c in ['since46', 'ever46', 'v2x_rule', 'v2x_polyarchy']], 2)
cov[:, :, 0] = np.log1p(cov[:, :, 0]); cov[:, :, 1] = np.log1p(cov[:, :, 1])
yearv = np.array([int(m[:4]) for m in months]); mi = {m: i for i, m in enumerate(months)}

def roll(x, k):                                   # sum over months t-k+1..t; nan until k months exist or if any is missing
    c = np.cumsum(np.nan_to_num(x), 1); out = c.copy(); out[:, k:] = c[:, k:] - c[:, :-k]; out[:, :k - 1] = np.nan
    bad = np.cumsum(np.isnan(x), 1); b = bad.copy(); b[:, k:] = bad[:, k:] - bad[:, :-k]
    out[b > 0] = np.nan; return out
s12 = roll(sb, 12)
at_risk = np.zeros((C, T), bool); at_risk[:, 23:] = (s12[:, 23:] < 25) & (s12[:, 11:-12] < 25)
def target(h):
    y = np.full((C, T), np.nan)
    for t in range(T - h):
        y[:, t] = (s12[:, t + 1:t + h + 1].max(1) >= 25).astype(float)
    y[~at_risk] = np.nan; return y
episode = np.zeros((C, T), bool); episode[:, 1:] = at_risk[:, :-1] & (s12[:, 1:] >= 25)      # month in which s12 first reaches 25
silent = at_risk & (s12 == 0)                                                                  # no state-based death in twelve months

# ------------------------------------------------------------------ features
L = np.log1p
U = np.stack([L(sb), L(roll(sb, 3)), L(s12), L(roll(sb, 24)), L(roll(ns + os_, 3)), L(roll(ns + os_, 12))], 2)
U = np.nan_to_num(U)
A = np.stack([L(a_pv), L(roll(a_pv, 3)), L(roll(a_pv, 12)), L(roll(a_fat, 3)), L(roll(a_fat, 12)), L(roll(a_demo, 3)), L(roll(a_demo, 12)),
              L(roll(a_pv, 3)) - L(roll(a_pv, 12) / 4)], 2)
F = np.stack([food / 100.0, roll(food, 3) / 300.0], 2); F = np.clip(F, -1, 5)
def fill(X):                                      # zero where a sensor is missing, with an indicator
    miss = np.isnan(X[:, :, :1]).astype(float); return np.concatenate([np.nan_to_num(X), miss], 2)
ALL_lin = np.concatenate([cov, U, fill(A), fill(F)], 2); ALL_gb = np.concatenate([cov, U, A, F], 2)

# ------------------------------------------------------------------ Tobit Kalman filter, several sensors
Yc = [L(sb), L(ns + os_), L(a_pv), L(a_fat), L(a_demo)]; Yf = np.clip(food / 100.0, -1, 5)
def unpack(th, K):
    phi = 1 / (1 + np.exp(-th[0])); q = np.exp(th[1]); a0 = th[2]; a = th[3:7]; o = 7
    c = np.concatenate([[0.0], th[o:o + K - 1]]); l = np.concatenate([[1.0], np.exp(th[o + K - 1:o + 2 * K - 2])]); r = np.exp(th[o + 2 * K - 2:o + 3 * K - 2])
    return phi, q, a0, a, c, l, r
def run(th, sensors, covz, t1, want=False):
    """Filter months 0..t1-1. Returns the negative log-likelihood, or filtered means and variances."""
    K = len(sensors); phi, q, a0, a, c, l, r = unpack(th, K)
    mu = a0 + covz @ a                                              # C x T
    m = mu[:, 0].copy(); P = np.full(C, q / max(1 - phi ** 2, 1e-3)); ll = 0.0
    M = np.zeros((C, t1)); V = np.zeros((C, t1))
    for t in range(t1):
        if t > 0:
            m = mu[:, t] + phi * (m - mu[:, t]); P = phi ** 2 * P + q
        for k, (kind, Y) in enumerate(sensors):
            y = Y[:, t]; ok = ~np.isnan(y)
            if not ok.any(): continue
            hm = c[k] + l[k] * m; S = l[k] ** 2 * P + r[k]
            pos = ok & ((y > 0) if kind == 'cens' else True); zero = ok & ~pos
            mn, Pn = m.copy(), P.copy()
            if pos.any():                                           # ordinary Kalman update
                Kg = l[k] * P[pos] / S[pos]; e = y[pos] - hm[pos]
                mn[pos] = m[pos] + Kg * e; Pn[pos] = (1 - Kg * l[k]) * P[pos]
                ll += np.sum(-0.5 * (np.log(2 * np.pi * S[pos]) + e ** 2 / S[pos]))
            if zero.any():                                          # censored reading: exact first two moments
                b = -l[k] / np.sqrt(r[k]); s = np.sqrt(1 + b ** 2 * P[zero]); eta = (-c[k] / np.sqrt(r[k]) + b * m[zero]) / s
                lam = np.exp(norm.logpdf(eta) - norm.logcdf(eta))
                mn[zero] = m[zero] + P[zero] * b * lam / s
                Pn[zero] = np.maximum(P[zero] - P[zero] ** 2 * b ** 2 / s ** 2 * lam * (eta + lam), 1e-6)
                ll += np.sum(norm.logcdf(eta))
            m, P = mn, Pn
        M[:, t] = m; V[:, t] = P
    return (M, V) if want else -ll
def fit_filter(sensors, covz, t1, th0, maxfun):
    f = lambda th: run(th, sensors, covz, t1) / 1e4
    res = minimize(f, th0, method='L-BFGS-B', options=dict(maxiter=200, maxfun=maxfun))
    return res.x

# ------------------------------------------------------------------ evaluation over windows
def auc(p, y, rows):
    k = rows & ~np.isnan(y) & ~np.isnan(p)
    return roc_auc_score(y[k], p[k]) if k.sum() > 0 and 0 < y[k].sum() < k.sum() else np.nan
def logit(X, y, tr):
    k = tr & ~np.isnan(y); sc = StandardScaler().fit(X[k]); mdl = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X[k]), y[k])
    return mdl.predict_proba(sc.transform(X.reshape(-1, X.shape[2])))[:, 1].reshape(C, T)
WINDOWS = [(2005, 2006, 2010), (2010, 2011, 2015), (2015, 2016, 2020), (2020, 2021, 2025)]
HOR = [3, 6, 12]; NAMES = ['L0 history', 'L1 + institutions', 'L2 + UCDP monthly deaths', 'F1 Tobit filter, deaths', 'F2 Tobit filter, six sensors', 'F3 Tobit filter, three death sensors', 'L3 logistic, all sensors', 'GB boosting, all sensors']
res = {h: {n: [] for n in NAMES} for h in HOR}; sil = {h: {n: [] for n in NAMES} for h in HOR}; acl = {h: {n: [] for n in NAMES} for h in HOR}
counts = []; keep = {}
th1 = None; th2 = None; th3 = None
for (tr_end, lo, hi) in WINDOWS:
    t1 = mi[f'{tr_end}-12'] + 1
    covm = np.nanmean(cov[:, :t1].reshape(-1, 4), 0); covs = np.nanstd(cov[:, :t1].reshape(-1, 4), 0)
    covz = np.nan_to_num((cov - covm) / covs)
    sens1 = [('cens', Yc[0])]; sens2 = [('cens', Yc[0]), ('cens', Yc[1]), ('cens', Yc[2]), ('cens', Yc[3]), ('cens', Yc[4]), ('lin', Yf)]
    first = th1 is None
    if first: th1 = np.array([2.5, -1.5, -4.0, 0, 0, 0, 0, 0.0])
    th1 = fit_filter(sens1, covz, t1, th1, 700 if first else 300)
    if th2 is None: th2 = np.concatenate([th1[:7], np.zeros(5), np.zeros(5), [th1[7]], np.zeros(5)])
    th2 = fit_filter(sens2, covz, t1, th2, 2000 if first else 800)
    sens3 = [('cens', Yc[0]), ('cens', Yc[1]), ('cens', Yc[3])]
    if th3 is None: th3 = np.concatenate([th1[:7], np.zeros(2), np.zeros(2), [th1[7]], np.zeros(2)])
    th3 = fit_filter(sens3, covz, t1, th3, 1000 if first else 400)
    M1, V1 = run(th1, sens1, covz, T, want=True); M2, V2 = run(th2, sens2, covz, T, want=True); M3, V3 = run(th3, sens3, covz, T, want=True)
    X = {'L0 history': covz[:, :, :2], 'L1 + institutions': covz, 'L2 + UCDP monthly deaths': np.concatenate([covz, U], 2),
         'F1 Tobit filter, deaths': np.concatenate([covz, M1[:, :, None], np.sqrt(V1)[:, :, None]], 2),
         'F2 Tobit filter, six sensors': np.concatenate([covz, M2[:, :, None], np.sqrt(V2)[:, :, None]], 2),
         'F3 Tobit filter, three death sensors': np.concatenate([covz, M3[:, :, None], np.sqrt(V3)[:, :, None]], 2),
         'L3 logistic, all sensors': np.concatenate([covz, ALL_lin[:, :, 4:]], 2)}
    te = np.zeros((C, T), bool); te[:, (yearv >= lo) & (yearv <= hi)] = True
    counts.append((lo, hi, int((episode & te).sum()), int((at_risk & te).sum()), int((silent & te).sum())))
    phi, q, a0, a, c, l, r = unpack(th2, 6)
    print(f'window {lo}-{hi}: fitted phi {phi:.3f} per month; loadings on the state: other deaths {l[1]:.2f}, ACLED events {l[2]:.2f}, ACLED fatalities {l[3]:.2f}, demonstrations {l[4]:.2f}, food inflation {l[5]:.3f}', flush=True)
    for h in HOR:
        y = target(h); tr = np.zeros((C, T), bool); tr[:, 24:t1 - h] = True; tr &= at_risk
        P = {n: logit(X[n], y, tr) for n in X}
        k = tr & ~np.isnan(y); live = ~np.all(np.isnan(ALL_gb[k]), 0)      # a sensor with no record yet in the fitting years is left out
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(ALL_gb[k][:, live], y[k])
        P['GB boosting, all sensors'] = gb.predict_proba(ALL_gb.reshape(-1, ALL_gb.shape[2])[:, live])[:, 1].reshape(C, T)
        for n in NAMES:
            res[h][n].append(auc(P[n], y, te & at_risk)); sil[h][n].append(auc(P[n], y, te & silent)); acl[h][n].append(auc(P[n], y, te & at_risk & ~np.isnan(a_pv)))
        if (lo, h) == (2021, 12): keep = dict(P=P, y=y, te=te)

print(f'\ncountries {C}, months {months[0]} to {months[-1]}; share of country-months with zero state-based deaths {np.mean(sb == 0):.3f}')
print('test window, onset episodes, at-risk country-months, of which silent (no state-based death in twelve months):')
for cnt in counts: print(f'   {cnt[0]}-{cnt[1]}: {cnt[2]:4d} {cnt[3]:7d} {cnt[4]:7d}')
def table(R, title):
    print('\n' + title); print(f'{"model":38s}' + ''.join(f'  h={h:<2d} mean (min-max)   ' for h in HOR))
    for n in NAMES:
        print(f'{n:38s}' + ''.join(f'  {np.nanmean(R[h][n]):.3f} ({np.nanmin(R[h][n]):.2f}-{np.nanmax(R[h][n]):.2f})     ' for h in HOR))
table(res, 'AUC for onset within h months, all at-risk country-months, mean over four windows')
table(sil, 'AUC among silent country-months (no state-based death in the last twelve months)')
table(acl, 'AUC among country-months inside ACLED coverage')
print('\nby window, horizon 12 months, all at-risk country-months:')
for n in NAMES: print(f'{n:38s}' + '  '.join(f'{v:.3f}' for v in res[12][n]))
# bootstrap by country, last window, horizon 12: six sensors against one, and against the regression on deaths
P, y, te = keep['P'], keep['y'], keep['te']; rows = te & at_risk & ~np.isnan(y)
def boot(a_, b_, n=1000):
    out = []
    for _ in range(n):
        cs = rng.integers(0, C, C); pa = P[a_][cs][rows[cs]]; pb = P[b_][cs][rows[cs]]; yy = y[cs][rows[cs]]
        if 0 < yy.sum() < len(yy): out.append(roc_auc_score(yy, pa) - roc_auc_score(yy, pb))
    return np.percentile(out, [2.5, 50, 97.5])
print('\nlast window, horizon 12 months, difference in AUC, bootstrap by country (2.5, 50, 97.5 percent):')
for a_, b_ in [('F3 Tobit filter, three death sensors', 'F1 Tobit filter, deaths'), ('F2 Tobit filter, six sensors', 'F1 Tobit filter, deaths'), ('F2 Tobit filter, six sensors', 'L2 + UCDP monthly deaths'), ('L2 + UCDP monthly deaths', 'L1 + institutions'), ('GB boosting, all sensors', 'L2 + UCDP monthly deaths')]:
    q_ = boot(a_, b_); print(f'   {a_}  minus  {b_}:  {q_[0]:+.3f}  {q_[1]:+.3f}  {q_[2]:+.3f}')
