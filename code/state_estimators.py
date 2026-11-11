"""
State estimators for the peace model: an observer for the violence state.

The model of the paper has no observer: the controller acts on the observation y as it
arrives. This script builds several observers from the shipped country-year panel and
tests them on the held-out years, with the same split as Table 3 / Table 11.2
(fit on rows 1991-2020, test on rows 2021-2024, whose outcomes are the onsets of 2022-2025).

State-space model, one latent state per country and year:
    z[t+1] = mu[t] + phi * (z[t] - mu[t]) + w,     w ~ N(0, q)        latent log intensity
    mu[t]  = a0 + a . covariates[t]                                   slow states set the level
    y[t]   = max(0, z[t] + v),                     v ~ N(0, r)        recorded log(1 + deaths)
The record is censored at zero: four fifths of country-years show no recorded deaths.

Estimators compared, all on the same information:
    logistic baselines (conflict history; history and deaths below the threshold; all inputs)
    Kalman filter         linear update, censoring ignored
    extended Kalman       softplus linearization of the censored measurement
    unscented Kalman      sigma points through max(0, z)
    particle filter       exact censored (Tobit) likelihood
    hidden Markov model   three regimes, Baum-Welch
    gradient boosting     scikit-learn HistGradientBoosting on lagged deaths and covariates
    neural network        feed-forward (scikit-learn MLP) on the same inputs
    Tobit Kalman filter   exact first two moments of the censored measurement
    RBF network, Haar-wavelet preprocessing + RBF network, recurrent high-order neural network
    (series-parallel, second-order terms), echo state network, random forest, hurdle model,
    Markov chain on death bands, and a mean-rank ensemble of all of them
    recurrent network     GRU over each country's history (PyTorch, if installed)

Environment variables TR_END, TE_LO, TE_HI set another window; code/estimator_ranking.py runs four.

Reads panels/panel2.csv. Run from the package root: python code/state_estimators.py
"""
import warnings, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
rng = np.random.default_rng(20261111)

# ------------------------------------------------------------------ data
p = pd.read_csv('panels/panel2.csv').sort_values(['country', 'year']).reset_index(drop=True)
countries = sorted(p.country.unique()); years = np.arange(1991, 2025)
C, T = len(countries), len(years)
ci = {c: i for i, c in enumerate(countries)}
def grid(col):
    g = np.full((C, T), np.nan)
    g[p.country.map(ci).values, (p.year - 1991).values] = p[col].astype(float).values
    return g
Y = grid('ldeaths'); Y0 = grid('ldeaths_prev')[:, 0]            # y in 1990, for initialization
ON, WAR = grid('onset'), grid('onset1000')                      # outcome in t+1, defined on at-risk rows
cov_names = ['since46', 'ever46', 'v2x_rule', 'v2x_polyarchy']
X = np.stack([grid(c) for c in cov_names], axis=2)
X[:, :, 0] = np.log1p(X[:, :, 0]); X[:, :, 1] = np.log1p(X[:, :, 1])
import os, json
TR_END = int(os.environ.get('TR_END', 2020)); TE_LO = int(os.environ.get('TE_LO', 2021)); TE_HI = int(os.environ.get('TE_HI', 2024))
FAST = os.environ.get('FAST', '0') == '1'
TR = years <= TR_END; TE = (years >= TE_LO) & (years <= TE_HI)       # rows; outcome is in year + 1
med = np.nanmedian(X[:, TR, :].reshape(-1, X.shape[2]), axis=0)
Xi = np.where(np.isnan(X), med, X)
mu_s, sd_s = Xi[:, TR, :].reshape(-1, 4).mean(0), Xi[:, TR, :].reshape(-1, 4).std(0)
Xs = (Xi - mu_s) / sd_s
THR25, THR1000 = np.log(26.0), np.log(1001.0)
print(f'countries {C}, years {years[0]}-{years[-1]}, share of country-years with zero recorded deaths {np.mean(Y == 0):.2f}')

# ------------------------------------------------------------------ filters
SOFT = 0.5
def softplus(z): return SOFT * np.logaddexp(0, z / SOFT)
def sigmoid(z): return 1 / (1 + np.exp(-z / SOFT))

def unpack(th):
    a0, a, phi = th[0], th[1:5], 1 / (1 + np.exp(-th[5]))
    return a0, a, phi, np.exp(th[6]), np.exp(th[7])

def run_filter(th, kind, Yobs=Y, y0=Y0, npart=400, return_all=False):
    """One pass over the years. Returns predictive mean and variance of z for year t+1 made at t,
    the filtered mean at t, and the censored predictive log-likelihood of y[t+1]."""
    a0, a, phi, q, r = unpack(th)
    mu = a0 + Xs @ a                                              # C x T
    zf = np.where(y0 > 0, y0, mu[:, 0] - 1.0); Pf = np.full(C, 1.0)
    if kind == 'pf':
        part = zf[:, None] + rng.normal(0, 1.0, (C, npart))
    ZP, PP, ZF, LL = np.zeros((C, T)), np.zeros((C, T)), np.zeros((C, T)), np.zeros((C, T))
    # the state at index t is the filtered state for year years[t], after seeing Y[:, t]
    for t in range(T):
        # predict year t from year t-1 (t = 0 is predicted from 1990)
        mprev = mu[:, t - 1] if t > 0 else mu[:, 0]
        if kind == 'pf':
            part = mprev[:, None] + phi * (part - mprev[:, None]) + rng.normal(0, np.sqrt(q), part.shape)
            zp, Pp = part.mean(1), part.var(1) + 1e-9
        else:
            zp, Pp = mprev + phi * (zf - mprev), phi ** 2 * Pf + q
        y = Yobs[:, t]; S = Pp + r
        ll = np.where(y > 0, norm.logpdf(y, zp, np.sqrt(S)), norm.logcdf(-zp / np.sqrt(S)))
        # update with y[t]
        if kind == 'kf':
            K = Pp / S; zf = zp + K * (y - zp); Pf = (1 - K) * Pp
        elif kind == 'ekf':
            H = sigmoid(zp); Se = H * Pp * H + r; K = Pp * H / Se
            zf = zp + K * (y - softplus(zp)); Pf = (1 - K * H) * Pp
        elif kind == 'ukf':
            lam = 2.0; s = np.sqrt((1 + lam) * Pp)
            sig = np.stack([zp, zp + s, zp - s], 1); w = np.array([lam / (1 + lam), 0.5 / (1 + lam), 0.5 / (1 + lam)])
            hs = np.maximum(sig, 0); yh = hs @ w
            Pyy = ((hs - yh[:, None]) ** 2) @ w + r; Pzy = ((sig - zp[:, None]) * (hs - yh[:, None])) @ w
            K = Pzy / Pyy; zf = zp + K * (y - yh); Pf = np.maximum(Pp - K * Pyy * K, 1e-6)
        elif kind == 'pf':
            lw = np.where(y[:, None] > 0, norm.logpdf(y[:, None], part, np.sqrt(r)), norm.logcdf(-part / np.sqrt(r)))
            w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            zf = (w * part).sum(1); Pf = (w * (part - zf[:, None]) ** 2).sum(1)
            cs = np.cumsum(w, 1); u = (rng.random((C, 1)) + np.arange(npart)[None, :]) / npart
            idx = (u[:, :, None] > cs[:, None, :]).sum(2).clip(0, npart - 1)
            part = np.take_along_axis(part, idx, 1)
        ZP[:, t], PP[:, t], ZF[:, t], LL[:, t] = zp, Pp, zf, ll
    # forecast for year t+1 made at year t: shift the predictions
    mu_t = mu
    if kind == 'pf':
        zn, Pn = mu_t + phi * (ZF - mu_t), None
    zn = mu_t + phi * (ZF - mu_t)
    return dict(zf=ZF, zpred_next=zn, ll=LL, phi=phi, q=q, r=r, mu=mu)

def negll(th):
    out = run_filter(th, 'ekf')
    return -out['ll'][:, TR].sum()

th0 = np.array([-2.0, -1.0, 1.0, -0.3, -0.3, 1.5, 0.5, 0.0])
res = minimize(negll, th0, method='L-BFGS-B')
th = res.x; a0, a, phi, q, r = unpack(th)
print('\n== State-space parameters (quasi-maximum likelihood on 1991-2020, censored likelihood)')
print(f'persistence phi {phi:.2f} per year; process variance q {q:.2f}; measurement variance r {r:.2f}')
print('level mu = %.2f ' % a0 + ' '.join(f'{v:+.2f}*{n}' for v, n in zip(a, ['log(1+years since conflict)', 'log(1+conflict years)', 'rule of law', 'electoral democracy'])) + '   (standardized inputs)')

def filter_forecast(kind):
    """Predictive distribution of z for year t+1, made with information through year t."""
    out = run_filter(th, kind)
    zf = out['zf']; mu = out['mu']
    # filtered variance is not stored for every method; rerun bookkeeping cheaply for the Gaussian ones
    zn = mu + phi * (zf - mu)
    return zf, zn

def filt_var(kind):
    a0_, a_, phi_, q_, r_ = unpack(th); mu = a0_ + Xs @ a_
    out = {}
    # second pass that also records the filtered variance
    zf = np.where(Y0 > 0, Y0, mu[:, 0] - 1.0); Pf = np.full(C, 1.0); PF = np.zeros((C, T)); ZF = np.zeros((C, T))
    if kind == 'pf':
        npart = 400; part = zf[:, None] + rng.normal(0, 1.0, (C, npart)); PARTN = np.zeros((C, T, npart))
    for t in range(T):
        mprev = mu[:, t - 1] if t > 0 else mu[:, 0]; y = Y[:, t]
        if kind == 'pf':
            part = mprev[:, None] + phi_ * (part - mprev[:, None]) + rng.normal(0, np.sqrt(q_), part.shape)
            lw = np.where(y[:, None] > 0, norm.logpdf(y[:, None], part, np.sqrt(r_)), norm.logcdf(-part / np.sqrt(r_)))
            w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            cs = np.cumsum(w, 1); u = (rng.random((C, 1)) + np.arange(npart)[None, :]) / npart
            idx = (u[:, :, None] > cs[:, None, :]).sum(2).clip(0, npart - 1)
            part = np.take_along_axis(part, idx, 1)
            ZF[:, t] = part.mean(1); PF[:, t] = part.var(1)
            PARTN[:, t, :] = mu[:, t][:, None] + phi_ * (part - mu[:, t][:, None]) + rng.normal(0, np.sqrt(q_), part.shape)
            continue
        zp, Pp = mprev + phi_ * (zf - mprev), phi_ ** 2 * Pf + q_; S = Pp + r_
        if kind == 'kf':
            K = Pp / S; zf = zp + K * (y - zp); Pf = (1 - K) * Pp
        elif kind == 'ekf':
            H = sigmoid(zp); Se = H * Pp * H + r_; K = Pp * H / Se; zf = zp + K * (y - softplus(zp)); Pf = (1 - K * H) * Pp
        elif kind == 'ukf':
            lam = 2.0; s = np.sqrt((1 + lam) * Pp); sig = np.stack([zp, zp + s, zp - s], 1)
            w = np.array([lam / (1 + lam), 0.5 / (1 + lam), 0.5 / (1 + lam)]); hs = np.maximum(sig, 0); yh = hs @ w
            Pyy = ((hs - yh[:, None]) ** 2) @ w + r_; Pzy = ((sig - zp[:, None]) * (hs - yh[:, None])) @ w
            K = Pzy / Pyy; zf = zp + K * (y - yh); Pf = np.maximum(Pp - K * Pyy * K, 1e-6)
        elif kind == 'tkf':
            sd = np.sqrt(S); pr = norm.cdf(zp / sd); ey = zp * pr + sd * norm.pdf(zp / sd)
            vy = np.maximum((zp ** 2 + S) * pr + zp * sd * norm.pdf(zp / sd) - ey ** 2, 1e-6)
            K = Pp * pr / vy; zf = zp + K * (y - ey); Pf = np.maximum(Pp - K * vy * K, 1e-6)
        ZF[:, t], PF[:, t] = zf, Pf
    if kind == 'pf':
        ynext = np.maximum(PARTN + rng.normal(0, np.sqrt(r_), PARTN.shape), 0)
        return ZF, dict(p25=(ynext >= THR25).mean(2), p1000=(ynext >= THR1000).mean(2), ymean=ynext.mean(2))
    zn = mu + phi_ * (ZF - mu); Sn = phi_ ** 2 * PF + q_ + r_; sd = np.sqrt(Sn)
    ymean = zn * norm.cdf(zn / sd) + sd * norm.pdf(zn / sd)       # mean of max(0, N(zn, Sn))
    return ZF, dict(p25=1 - norm.cdf((THR25 - zn) / sd), p1000=1 - norm.cdf((THR1000 - zn) / sd), ymean=ymean)

# ------------------------------------------------------------------ hidden Markov model, three regimes
def hmm_fit(Yseq, n_iter=60):
    K = 3
    A = np.array([[.97, .025, .005], [.15, .75, .10], [.03, .17, .80]]); pi = np.array([.8, .15, .05])
    p0 = np.array([.97, .25, .02]); m = np.array([1.5, 3.5, 7.0]); s = np.array([1.0, 1.2, 1.2])
    for _ in range(n_iter):
        B = np.where(Yseq[:, :, None] == 0, p0, (1 - p0) * norm.pdf(Yseq[:, :, None], m, s)) + 1e-300
        n, Tn = Yseq.shape
        al = np.zeros((n, Tn, K)); sc = np.zeros((n, Tn)); be = np.ones((n, Tn, K))
        al[:, 0] = pi * B[:, 0]; sc[:, 0] = al[:, 0].sum(1); al[:, 0] /= sc[:, [0]]
        for t in range(1, Tn):
            al[:, t] = (al[:, t - 1] @ A) * B[:, t]; sc[:, t] = al[:, t].sum(1); al[:, t] /= sc[:, [t]]
        for t in range(Tn - 2, -1, -1):
            be[:, t] = (be[:, t + 1] * B[:, t + 1]) @ A.T / sc[:, [t + 1]]
        g = al * be; g /= g.sum(2, keepdims=True)
        xi = al[:, :-1, :, None] * A[None, None] * (B[:, 1:] * be[:, 1:])[:, :, None, :] / sc[:, 1:, None, None]
        A = xi.sum((0, 1)); A /= A.sum(1, keepdims=True); pi = g[:, 0].mean(0)
        z0 = (Yseq == 0)[:, :, None]
        p0 = np.clip((g * z0).sum((0, 1)) / g.sum((0, 1)), 1e-3, 1 - 1e-3)
        wpos = g * (~z0); m = (wpos * Yseq[:, :, None]).sum((0, 1)) / wpos.sum((0, 1))
        s = np.sqrt((wpos * (Yseq[:, :, None] - m) ** 2).sum((0, 1)) / wpos.sum((0, 1))) + 1e-3
        o = np.argsort(m); A, pi, p0, m, s = A[o][:, o], pi[o], p0[o], m[o], s[o]
    return A, pi, p0, m, s

def hmm_forecast(Yall, pars):
    A, pi, p0, m, s = pars
    B = np.where(Yall[:, :, None] == 0, p0, (1 - p0) * norm.pdf(Yall[:, :, None], m, s)) + 1e-300
    n, Tn = Yall.shape; al = np.zeros((n, Tn, 3))
    al[:, 0] = pi * B[:, 0]; al[:, 0] /= al[:, [0]].sum(2)
    for t in range(1, Tn):
        al[:, t] = (al[:, t - 1] @ A) * B[:, t]; al[:, t] /= al[:, t].sum(1, keepdims=True)
    nxt = al @ A
    f = lambda thr: (nxt * (1 - p0) * (1 - norm.cdf((thr - m) / s))).sum(2)
    ymean = (nxt * (1 - p0) * m).sum(2)
    return al, dict(p25=f(THR25), p1000=f(THR1000), ymean=ymean)

# ------------------------------------------------------------------ learned estimators
def lag(G, k):
    out = np.full_like(G, 0.0); out[:, k:] = G[:, :-k]
    if k == 1: out[:, 0] = Y0
    return out
feat = np.stack([Y, lag(Y, 1), lag(Y, 2), Xi[:, :, 0], Xi[:, :, 1], grid('v2x_rule'), grid('v2x_polyarchy'),
                 grid('v2x_corr'), grid('v2x_freexp_altinf'), grid('mil')], 2)
feat_i = np.where(np.isnan(feat), np.nanmedian(feat[:, TR].reshape(-1, feat.shape[2]), 0), feat)
Ynext = np.full((C, T), np.nan); Ynext[:, :-1] = Y[:, 1:]

def fit_predict_tabular(make, target, proba=True, impute=False):
    F = feat_i if impute else feat
    m_tr = TR[None, :] & ~np.isnan(target); m_tr[:, ~TR] = False
    Xtr = F[m_tr]; ytr = target[m_tr]
    if impute:
        sc = StandardScaler().fit(Xtr); Xtr = sc.transform(Xtr); Fall = sc.transform(F.reshape(-1, F.shape[2]))
    else:
        Fall = F.reshape(-1, F.shape[2])
    preds = np.zeros(C * T)
    for seed in range(5):
        mdl = make(seed).fit(Xtr, ytr)
        preds += (mdl.predict_proba(Fall)[:, 1] if proba else mdl.predict(Fall)) / 5
    return preds.reshape(C, T)

def gru_predict():
    try:
        import torch, torch.nn as nn
    except Exception:
        return None
    torch.manual_seed(0)
    Fz = torch.tensor(StandardScaler().fit(feat_i[:, TR].reshape(-1, feat_i.shape[2])).transform(feat_i.reshape(-1, feat_i.shape[2])).reshape(C, T, -1), dtype=torch.float32)
    tgt = torch.tensor(np.nan_to_num(np.stack([ON, WAR], 2)), dtype=torch.float32)
    msk = torch.tensor((~np.isnan(np.stack([ON, WAR], 2))) & TR[None, :, None], dtype=torch.float32)
    out = np.zeros((C, T, 2))
    for seed in range(5):
        torch.manual_seed(seed)
        g = nn.GRU(Fz.shape[2], 12, batch_first=True); h = nn.Linear(12, 2)
        opt = torch.optim.Adam(list(g.parameters()) + list(h.parameters()), lr=0.01, weight_decay=1e-3)
        pw = torch.tensor([40.0, 60.0])
        for ep in range(150):
            opt.zero_grad(); o, _ = g(Fz); lo = h(o)
            loss = (nn.functional.binary_cross_entropy_with_logits(lo, tgt, reduction='none', pos_weight=pw) * msk).sum() / msk.sum()
            loss.backward(); opt.step()
        with torch.no_grad():
            o, _ = g(Fz); out += torch.sigmoid(h(o)).numpy() / 5
    return out

# ------------------------------------------------------------------ evaluation
def metrics(prob, target, rows):
    m = rows[None, :] & ~np.isnan(target); y = target[m]; s = prob[m]
    auc = roc_auc_score(y, s)
    k = max(1, int(round(0.1 * len(y)))); top = np.argsort(-s)[:k]
    return auc, int(y[top].sum()), int(y.sum()), len(y), m

def boot_auc(prob, target, rows, base=None, n=2000):
    m = rows[None, :] & ~np.isnan(target); cc = np.where(m.any(1))[0]; out = []; dif = []
    for _ in range(n):
        pick = rng.choice(cc, len(cc)); yy = np.concatenate([target[c, rows & ~np.isnan(target[c])] for c in pick])
        ss = np.concatenate([prob[c, rows & ~np.isnan(target[c])] for c in pick])
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        out.append(roc_auc_score(yy, ss))
        if base is not None:
            bb = np.concatenate([base[c, rows & ~np.isnan(target[c])] for c in pick]); dif.append(out[-1] - roc_auc_score(yy, bb))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return (lo, hi) if base is None else (lo, hi, np.percentile(dif, 2.5), np.percentile(dif, 97.5))

def logit_scores(cols, target):
    F = np.stack(cols, 2); m = TR[None, :] & ~np.isnan(target)
    mdl = LogisticRegression(max_iter=2000).fit(F[m], target[m])
    return mdl.predict_proba(F.reshape(-1, F.shape[2]))[:, 1].reshape(C, T)

since46, ever46 = grid('since46'), grid('ever46')
scores = {}
scores['Logistic, conflict history (baseline)'] = dict(p25=logit_scores([since46, ever46], ON), p1000=logit_scores([since46, ever46, Y, lag(Y, 1)], WAR), ymean=None)
scores['Logistic, same inputs as the filters'] = dict(p25=logit_scores([Xs[:, :, 0], Xs[:, :, 1], Xs[:, :, 2], Xs[:, :, 3], Y, lag(Y, 1)], ON), p1000=logit_scores([Xs[:, :, 0], Xs[:, :, 1], Xs[:, :, 2], Xs[:, :, 3], Y, lag(Y, 1)], WAR), ymean=None)
scores['Logistic, history and recorded deaths below the threshold'] = dict(p25=logit_scores([since46, ever46, Y, lag(Y, 1)], ON), p1000=logit_scores([since46, ever46, Y, lag(Y, 1)], WAR), ymean=None)
ZFs = {}
for kind, name in [('kf', 'Kalman filter'), ('ekf', 'Extended Kalman filter'), ('ukf', 'Unscented Kalman filter'), ('tkf', 'Tobit Kalman filter'), ('pf', 'Particle filter (censored likelihood)')]:
    ZFs[kind], scores[name] = filt_var(kind)
hp = hmm_fit(np.concatenate([Y0[:, None], Y[:, TR]], 1))
_, sc_h = hmm_forecast(np.concatenate([Y0[:, None], Y], 1), hp)
scores['Hidden Markov model, three regimes'] = {k: v[:, 1:] for k, v in sc_h.items()}
gb = lambda seed: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, l2_regularization=1.0, random_state=seed)
gbr = lambda seed: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, random_state=seed)
scores['Gradient boosting'] = dict(p25=fit_predict_tabular(gb, ON), p1000=fit_predict_tabular(gb, WAR), ymean=fit_predict_tabular(gbr, Ynext, proba=False))
mlp = lambda seed: MLPClassifier(hidden_layer_sizes=(16, 8), alpha=1e-2, max_iter=800, random_state=seed)
scores['Neural network (feed-forward)'] = dict(p25=fit_predict_tabular(mlp, ON, impute=True), p1000=fit_predict_tabular(mlp, WAR, impute=True), ymean=None)

# ------------------------------------------------------------------ further estimators
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
SCL = StandardScaler().fit(feat_i[:, TR].reshape(-1, feat_i.shape[2]))
Fz = SCL.transform(feat_i.reshape(-1, feat_i.shape[2])).reshape(C, T, -1)

def readout(Z, target, Cc=1.0):
    """Logistic readout on a feature array Z (C x T x k), fitted on the training rows."""
    m = TR[None, :] & ~np.isnan(target)
    sc = StandardScaler().fit(Z[m]); mdl = LogisticRegression(max_iter=4000, C=Cc).fit(sc.transform(Z[m]), target[m])
    return mdl.predict_proba(sc.transform(Z.reshape(-1, Z.shape[2])))[:, 1].reshape(C, T)

def rbf_features(Z, k=40):
    flat = Z[:, TR].reshape(-1, Z.shape[2]); km = KMeans(k, n_init=5, random_state=0).fit(flat); cen = km.cluster_centers_
    dc = np.sqrt(((cen[:, None, :] - cen[None, :, :]) ** 2).sum(2)); sig = np.median(dc[dc > 0]) / np.sqrt(2)
    d2 = ((Z.reshape(-1, 1, Z.shape[2]) - cen[None]) ** 2).sum(2)
    return np.concatenate([np.exp(-d2 / (2 * sig ** 2)).reshape(C, T, k), Z], 2)

# radial-basis-function network
RB = rbf_features(Fz)
scores['Radial-basis-function network'] = dict(p25=readout(RB, ON), p1000=readout(RB, WAR), ymean=None)

# Haar wavelet preprocessing of the last eight years of the death series, then the RBF network
def haar8(Yg):
    pad = np.concatenate([np.repeat(Y0[:, None], 7, 1), Yg], 1); out = np.zeros((C, T, 8))
    for t in range(T):
        w = pad[:, t:t + 8]; a = w; co = []
        for _ in range(3):
            d_ = (a[:, 0::2] - a[:, 1::2]) / np.sqrt(2); a = (a[:, 0::2] + a[:, 1::2]) / np.sqrt(2); co.append(d_)
        out[:, t] = np.concatenate([a] + co[::-1], 1)
    return out
WV = np.concatenate([haar8(Y), Fz[:, :, 3:]], 2)
WVz = StandardScaler().fit(WV[:, TR].reshape(-1, WV.shape[2])).transform(WV.reshape(-1, WV.shape[2])).reshape(C, T, -1)
RBW = rbf_features(WVz)
scores['Wavelet preprocessing + RBF network'] = dict(p25=readout(RBW, ON), p1000=readout(RBW, WAR), ymean=None)

# recurrent high-order neural network, series-parallel form: first- and second-order sigmoid terms
sg = 1 / (1 + np.exp(-Fz)); iu = np.triu_indices(Fz.shape[2])
HO = np.concatenate([sg, (sg[:, :, :, None] * sg[:, :, None, :])[:, :, iu[0], iu[1]]], 2)
scores['Recurrent high-order neural network'] = dict(p25=readout(HO, ON, Cc=0.3), p1000=readout(HO, WAR, Cc=0.3), ymean=None)

# echo state network (reservoir computing)
def esn_scores(target, n=60, seeds=5):
    out = np.zeros((C, T))
    for sd_ in range(seeds):
        r_ = np.random.default_rng(100 + sd_); Wr = r_.normal(0, 1, (n, n)) * (r_.random((n, n)) < 0.2)
        Wr *= 0.7 / max(abs(np.linalg.eigvals(Wr))); Wi = r_.normal(0, 0.5, (n, Fz.shape[2]))
        h = np.zeros((C, n)); H = np.zeros((C, T, n))
        for t in range(T):
            h = np.tanh(h @ Wr.T + Fz[:, t] @ Wi.T); H[:, t] = h
        out += readout(np.concatenate([H, Fz], 2), target, Cc=0.1) / seeds
    return out
scores['Echo state network'] = dict(p25=esn_scores(ON), p1000=esn_scores(WAR), ymean=None)

# random forest
rf = lambda seed: RandomForestClassifier(n_estimators=400, min_samples_leaf=15, random_state=seed, n_jobs=2)
scores['Random forest'] = dict(p25=fit_predict_tabular(rf, ON, impute=True), p1000=fit_predict_tabular(rf, WAR, impute=True), ymean=None)

# hurdle model: any recorded deaths next year, then their size
mh = TR[None, :] & ~np.isnan(Ynext); pos = mh & (Ynext > 0)
hl = LogisticRegression(max_iter=4000).fit(Fz[mh], (Ynext[mh] > 0).astype(int)); hr = LinearRegression().fit(Fz[pos], Ynext[pos])
ppos = hl.predict_proba(Fz.reshape(-1, Fz.shape[2]))[:, 1].reshape(C, T); mpos = hr.predict(Fz.reshape(-1, Fz.shape[2])).reshape(C, T)
spos = np.std(Ynext[pos] - hr.predict(Fz[pos]))
scores['Hurdle model'] = dict(p25=ppos * (1 - norm.cdf((THR25 - mpos) / spos)), p1000=ppos * (1 - norm.cdf((THR1000 - mpos) / spos)), ymean=ppos * mpos)

# Markov chain on death bands of the two preceding years
band = lambda G: np.digitize(np.expm1(G), [0.5, 24.5, 999.5]); bs = band(Y) * 4 + band(lag(Y, 1))
def markov(target):
    m = TR[None, :] & ~np.isnan(target); base = target[m].mean(); out = np.full((C, T), base)
    for k in range(16):
        sel = m & (bs == k); n_ = sel.sum()
        out[bs == k] = (target[sel].sum() + 5 * base) / (n_ + 5)
    return out
scores['Markov chain on death bands'] = dict(p25=markov(ON), p1000=markov(WAR), ymean=None)

# ensemble: mean rank of every estimator that uses the full inputs
from scipy.stats import rankdata
def ensemble(key, target):
    m = ~np.isnan(target); out = np.zeros((C, T)); k = 0
    for name, sc_ in scores.items():
        if name.startswith('Logistic, conflict history') or name.startswith('Ensemble'): continue
        r_ = np.zeros((C, T)); r_[m] = rankdata(sc_[key][m]) / m.sum(); out += r_; k += 1
    return out / k
g = gru_predict()
if g is not None:
    scores['Recurrent network (GRU)'] = dict(p25=g[:, :, 0], p1000=g[:, :, 1], ymean=None)
else:
    print('\n(PyTorch not installed: the gated recurrent network is skipped)')
scores['Ensemble (mean rank of all estimators)'] = dict(p25=ensemble('p25', ON), p1000=ensemble('p1000', WAR), ymean=None)

print(f'\n== Held-out test: fit on rows through {TR_END}; test rows {TE_LO}-{TE_HI}, outcomes {TE_LO+1}-{TE_HI+1}')
print(f"{'estimator':58s} {'onset AUC':>10s} {'95% interval':>14s} {'top tenth':>10s} {'gain over baseline (95%)':>26s} {'war AUC':>8s} {'top tenth':>10s}")
base25 = scores['Logistic, conflict history (baseline)']['p25']
table = []
for name, s in scores.items():
    a1, k1, n1, N1, _ = metrics(s['p25'], ON, TE); a2, k2, n2, N2, _ = metrics(s['p1000'], WAR, TE)
    if FAST:
        lo = hi = float('nan'); gain = ''
    elif name.startswith('Logistic, conflict history'):
        lo, hi = boot_auc(s['p25'], ON, TE); gain = ''
    else:
        lo, hi, dlo, dhi = boot_auc(s['p25'], ON, TE, base=base25); gain = f'{a1 - metrics(base25, ON, TE)[0]:+.3f} ({dlo:+.2f} to {dhi:+.2f})'
    print(f'{name:58s} {a1:10.3f} {f"{lo:.2f}-{hi:.2f}":>14s} {f"{k1} of {n1}":>10s} {gain:>26s} {a2:8.3f} {f"{k2} of {n2}":>10s}')
    table.append((name, a1, lo, hi, k1, n1, a2, k2, n2))
print(f'test rows: onset {N1} with {n1} onsets; war {N2} with {n2} escalations')
if os.environ.get('DUMP'):
    json.dump([dict(name=t_[0], auc=t_[1], lo=t_[2], hi=t_[3], top=t_[4], n=t_[5], war=t_[6], wtop=t_[7], wn=t_[8]) for t_ in table], open(os.environ['DUMP'], 'w'))
if FAST: sys.exit(0)

mm = TE[None, :] & ~np.isnan(ON); low = (Y > 0) | (lag(Y, 1) > 0)
print(f'at-risk test rows with some recorded deaths below the threshold in t or t-1: {int((mm & low).sum())} of {int(mm.sum())}; onsets among them {int(ON[mm & low].sum())} of {int(ON[mm].sum())}')
mt = TR[None, :] & ~np.isnan(ON)
print(f'same, fit rows: {int((mt & low).sum())} of {int(mt.sum())}; onsets among them {int(ON[mt & low].sum())} of {int(ON[mt].sum())}')
# onsets from recorded silence: no recorded deaths in t and t-1
sil = (Y == 0) & (lag(Y, 1) == 0)
print('\n== Onsets from recorded silence (zero recorded deaths in the year and the year before), test rows')
for name, s in scores.items():
    tgt = np.where(sil, ON, np.nan); m = TE[None, :] & ~np.isnan(tgt)
    print(f'{name:40s} AUC {roc_auc_score(tgt[m], s["p25"][m]):.3f}   ({int(tgt[m].sum())} onsets in {int(m.sum())} rows)')

print('\n== One-year-ahead prediction of log(1 + deaths), test rows 2021-2023 (target years 2022-2024), all countries')
m = np.zeros((C, T), bool); m[:, (years >= 2021) & (years <= 2023)] = True
print(f"{'persistence (next year = this year)':40s} RMSE {np.sqrt(np.mean((Y[m] - Ynext[m]) ** 2)):.3f}")
for name, s in scores.items():
    if s.get('ymean') is not None:
        print(f'{name:40s} RMSE {np.sqrt(np.mean((s["ymean"][m] - Ynext[m]) ** 2)):.3f}')

# ------------------------------------------------------------------ known-truth test
print('\n== Known-truth test: data simulated from the fitted state-space model, 20 replications')
print('RMSE of the filtered latent state against the true state; "silent" = years with no recorded deaths')
mu = a0 + Xs @ a; acc = {k: [] for k in ['kf', 'ekf', 'ukf', 'tkf', 'pf']}; accs = {k: [] for k in acc}; hid = {k: [] for k in acc}
Ysave, Y0save = Y.copy(), Y0.copy()
for rep in range(20):
    z = np.zeros((C, T)); zprev = mu[:, 0] + rng.normal(0, np.sqrt(q / (1 - phi ** 2)), C)
    y0 = np.maximum(zprev + rng.normal(0, np.sqrt(r), C), 0)
    for t in range(T):
        mp = mu[:, t - 1] if t > 0 else mu[:, 0]
        z[:, t] = mp + phi * (zprev - mp) + rng.normal(0, np.sqrt(q), C); zprev = z[:, t]
    ysim = np.maximum(z + rng.normal(0, np.sqrt(r), (C, T)), 0)
    Y, Y0 = ysim, y0
    for k in acc:
        zf, sc = filt_var(k)
        acc[k].append(np.sqrt(np.mean((zf - z) ** 2))); accs[k].append(np.sqrt(np.mean((zf - z)[ysim == 0] ** 2)))
        # does the estimate rank next-year onsets among silent rows?  truth-based ceiling is reported once
        tgt = (ysim[:, 1:] >= THR25) & (ysim[:, :-1] == 0)
        base_rows = ysim[:, :-1] == 0
        hid[k].append(roc_auc_score(tgt[base_rows], sc['p25'][:, :-1][base_rows]))
    if rep == 0: ceil = []
    ztrue_next = mu[:, :-1] + phi * (z[:, :-1] - mu[:, :-1])
    ceil.append(roc_auc_score(tgt[base_rows], ztrue_next[base_rows]))
Y, Y0 = Ysave, Y0save
for k, name in [('kf', 'Kalman filter'), ('ekf', 'Extended Kalman filter'), ('ukf', 'Unscented Kalman filter'), ('tkf', 'Tobit Kalman filter'), ('pf', 'Particle filter (censored likelihood)')]:
    print(f'{name:40s} RMSE all years {np.mean(acc[k]):.3f}   silent years {np.mean(accs[k]):.3f}   AUC for onset from silence {np.mean(hid[k]):.3f}')
print(f"{'true state known (ceiling)':40s} {'':47s} AUC for onset from silence {np.mean(ceil):.3f}")
print(f'prior standard deviation of the latent state: {np.sqrt(q / (1 - phi ** 2)):.3f}')
