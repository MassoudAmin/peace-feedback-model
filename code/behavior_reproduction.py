"""
Behavior-reproduction test of the fitted violence equation.

The one equation of the model with a fitted form is the violence state: a latent log intensity z per
country that reverts at rate phi toward a level set by conflict history, rule of law, and electoral
democracy, recorded as y = max(0, z + v). It is fitted on 1991-2020 by censored quasi-maximum
likelihood, exactly as in code/state_estimators.py. This script simulates 500 panels from the fitted
equation, with each country's own covariates and its 1990 record as the starting point, and compares
statistics of the simulated panels with the same statistics of the record. Two comparisons are made:
the fit period 1991-2020, and the held-out years 2021-2024 simulated forward from the 2020 state
(parameters from 1991-2020 only). The fear, reserve, and arms equations are not fitted: no
cross-country series exists for them.

Reads panels/panel2.csv. Run from the package root: python code/behavior_reproduction.py
"""
import warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
warnings.filterwarnings('ignore')
rng = np.random.default_rng(20261112)
p = pd.read_csv('panels/panel2.csv').sort_values(['country', 'year']).reset_index(drop=True)
countries = sorted(p.country.unique()); years = np.arange(1991, 2025)
C, T = len(countries), len(years); ci = {c: i for i, c in enumerate(countries)}
def grid(col):
    g = np.full((C, T), np.nan)
    g[p.country.map(ci).values, (p.year - 1991).values] = p[col].astype(float).values
    return g
Y = grid('ldeaths'); Y0 = grid('ldeaths_prev')[:, 0]
X = np.stack([grid(c) for c in ['since46', 'ever46', 'v2x_rule', 'v2x_polyarchy']], axis=2)
X[:, :, 0] = np.log1p(X[:, :, 0]); X[:, :, 1] = np.log1p(X[:, :, 1])
TR = years <= 2020; TE = years >= 2021
med = np.nanmedian(X[:, TR, :].reshape(-1, 4), axis=0); Xi = np.where(np.isnan(X), med, X)
Xs = (Xi - Xi[:, TR, :].reshape(-1, 4).mean(0)) / Xi[:, TR, :].reshape(-1, 4).std(0)
Y = np.where(np.isnan(Y), 0.0, Y); Y0 = np.where(np.isnan(Y0), 0.0, Y0)
SOFT = 0.5
softplus = lambda z: SOFT * np.logaddexp(0, z / SOFT); sigmoid = lambda z: 1 / (1 + np.exp(-z / SOFT))
def unpack(th): return th[0], th[1:5], 1 / (1 + np.exp(-th[5])), np.exp(th[6]), np.exp(th[7])
def ekf(th):
    a0, a, phi, q, r = unpack(th); mu = a0 + Xs @ a
    zf = np.where(Y0 > 0, Y0, mu[:, 0] - 1.0); Pf = np.full(C, 1.0); LL = np.zeros((C, T)); ZF = np.zeros((C, T)); PF = np.zeros((C, T))
    for t in range(T):
        mp = mu[:, t - 1] if t > 0 else mu[:, 0]
        zp, Pp = mp + phi * (zf - mp), phi ** 2 * Pf + q
        y = Y[:, t]; S = Pp + r
        LL[:, t] = np.where(y > 0, norm.logpdf(y, zp, np.sqrt(S)), norm.logcdf(-zp / np.sqrt(S)))
        H = sigmoid(zp); Se = H * Pp * H + r; K = Pp * H / Se
        zf = zp + K * (y - softplus(zp)); Pf = (1 - K * H) * Pp; ZF[:, t] = zf; PF[:, t] = Pf
    return LL, ZF, PF, mu
res = minimize(lambda th: -ekf(th)[0][:, TR].sum(), np.array([-2.0, -1.0, 1.0, -0.3, -0.3, 1.5, 0.5, 0.0]), method='L-BFGS-B')
a0, a, phi, q, r = unpack(res.x); LL, ZF, PF, mu = ekf(res.x)
print(f'fitted on 1991-2020: phi {phi:.2f} per year, process variance {q:.2f}, measurement variance {r:.2f}')
L25, L100, L500, L1000 = np.log(26.0), np.log(101.0), np.log(501.0), np.log(1001.0)

def stats(y, yprev0):
    """y: C x T' panel of log(1+deaths); yprev0: the year before the first column."""
    yl1 = np.concatenate([yprev0[:, None], y[:, :-1]], 1)                  # t-1
    yl2 = np.concatenate([yprev0[:, None], yl1[:, :-1]], 1)                # t-2 (first column repeats)
    s = {}
    s['share of country-years with zero recorded deaths'] = np.mean(y == 0)
    on = (y >= L25) & (yl1 < L25) & (yl2 < L25)
    s['new conflicts (25 or more after two years below)'] = on.sum()
    s['share of new conflicts that follow a year with zero deaths'] = (on & (yl1 == 0)).sum() / max(on.sum(), 1)
    war = (y >= L1000) & (yl1 < L1000) & (yl2 < L1000)
    s['war onsets (1,000 or more after two years below)'] = war.sum()
    s['country-years at 1,000 or more deaths'] = (y >= L1000).sum()
    for lo, hi, nm in [(0, 1e-9, '0'), (1e-9, L25, '1-24'), (L25, L100, '25-99'), (L100, L500, '100-499'), (L500, L1000, '500-999')]:
        m = (yl1 >= lo) & (yl1 < hi) if lo > 0 else (yl1 == 0)
        s[f'P(1,000 or more next year | {nm} this year), percent'] = 100 * np.mean(y[m] >= L1000) if m.sum() else np.nan
    pos = (y > 0) & (yl1 > 0)
    s['year-to-year correlation of log deaths, active years'] = np.corrcoef(y[pos], yl1[pos])[0, 1] if pos.sum() > 5 else np.nan
    s['countries with any year at 25 or more deaths'] = (y >= L25).any(1).sum()
    return s

def simulate(z0, t0, t1, n=500):
    out = []
    for _ in range(n):
        z = np.zeros((C, t1 - t0)); zprev = z0()
        for k, t in enumerate(range(t0, t1)):
            mp = mu[:, t - 1] if t > 0 else mu[:, 0]
            z[:, k] = mp + phi * (zprev - mp) + rng.normal(0, np.sqrt(q), C); zprev = z[:, k]
        out.append(np.maximum(z + rng.normal(0, np.sqrt(r), z.shape), 0))
    return out

def compare(title, yrec, yprev, sims):
    rec = stats(yrec, yprev); S = [stats(y, yprev) for y in sims]
    print(f'\n== {title}')
    print(f"{'statistic':62s} {'record':>9s} {'simulated median':>17s} {'10-90 percent':>16s}  inside")
    inside = 0
    for k in rec:
        v = np.array([s[k] for s in S], float); lo, md, hi = np.nanpercentile(v, [10, 50, 90])
        ok = lo <= rec[k] <= hi; inside += ok
        f = (lambda x: f'{x:9.0f}') if rec[k] > 20 and float(rec[k]).is_integer() else (lambda x: f'{x:9.2f}')
        print(f'{k:62s} {f(rec[k])} {f(md):>17s} {f(lo).strip():>7s} to {f(hi).strip():<7s} {"yes" if ok else "NO"}')
    print(f'{inside} of {len(rec)} record statistics lie inside the simulated 10-90 percent band')

# fit period: start every country from its 1990 record
z1990 = lambda: np.where(Y0 > 0, Y0, mu[:, 0] - 1.0) + rng.normal(0, 1.0, C)
compare('Fit period 1991-2020: 500 panels simulated from the fitted equation', Y[:, TR], Y0, simulate(z1990, 0, int(TR.sum())))
# held-out years: start from the filtered 2020 state, parameters from 1991-2020 only
i20 = int(TR.sum()) - 1
z2020 = lambda: ZF[:, i20] + rng.normal(0, 1, C) * np.sqrt(PF[:, i20])
compare('Held-out years 2021-2024: 500 panels simulated forward from the 2020 state', Y[:, TE], Y[:, i20], simulate(z2020, i20 + 1, T))

# ------------------------------------------------------------------ second form: persistence that depends on the state
# The first form misses onset from silence, the length of wars, and persistence in active years. The
# second form lets the equation differ between a quiet year and an active one, and gives the shock a
# heavy tail:   y*[t] = a0 + X a + 1[y[t-1] > 0] (A + rho y[t-1]) + e,   y[t] = y* if y* >= c0, else 0,
# with e drawn from a mixture of two normal distributions and c0 = log(1.5), the level below which the
# count rounds to no deaths. It is fitted on 1991-2020 by maximum likelihood (the censored likelihood is
# exact here), not by matching the statistics below, and it is then judged on the same twelve statistics.
C0 = np.log(1.5)
YL = np.concatenate([Y0[:, None], Y[:, :-1]], 1)
def parts(th):
    return th[0], th[1:5], th[5], 1 / (1 + np.exp(-th[6])), np.exp(th[7]), 1 / (1 + np.exp(-th[8])), th[9], np.exp(th[10])
def mean2(th, ylag, t):
    b0, b, A, rho, s1, w, m2, s2 = parts(th)
    return b0 + Xs[:, t, :] @ b + (ylag > 0) * (A + rho * ylag)
def nll2(th):
    b0, b, A, rho, s1, w, m2, s2 = parts(th); tot = 0.0
    for t in np.where(TR)[0]:
        m = mean2(th, YL[:, t], t); y = Y[:, t]
        dens = (1 - w) * norm.pdf(y, m, s1) + w * norm.pdf(y, m + m2, s2)
        cens = (1 - w) * norm.cdf(C0, m, s1) + w * norm.cdf(C0, m + m2, s2)
        tot += np.sum(np.where(y > 0, np.log(dens + 1e-300), np.log(cens + 1e-300)))
    return -tot
best = None
for start in [[-3, -.5, .5, -.3, -.3, 1.0, 1.0, 0.3, -2.0, 4.0, 0.5], [-4, -.5, .5, -.3, -.3, 2.0, 1.5, 0.0, -3.0, 6.0, 0.0], [-2, 0, 0, 0, 0, 0.5, 0.5, 0.5, -1.0, 2.0, 0.7]]:
    r2 = minimize(nll2, np.array(start, float), method='L-BFGS-B')
    if best is None or r2.fun < best.fun: best = r2
th2 = best.x; b0, b, A, rho, s1, w, m2, s2 = parts(th2)
print(f'\nsecond form, fitted on 1991-2020 by maximum likelihood: persistence in active years rho {rho:.2f}; shift when active {A:+.2f}; '
      f'shock: {100 * (1 - w):.0f} percent sd {s1:.2f}, {100 * w:.0f} percent mean {m2:+.2f} sd {s2:.2f}; log-likelihood {-best.fun:.0f}')
def simulate2(ystart, t0, t1, n=500):
    out = []
    for _ in range(n):
        y = np.zeros((C, t1 - t0)); yl = ystart.copy()
        for k, t in enumerate(range(t0, t1)):
            big = rng.random(C) < w
            e = np.where(big, m2 + s2 * rng.normal(size=C), s1 * rng.normal(size=C))
            ys = mean2(th2, yl, t) + e
            y[:, k] = np.where(ys >= C0, ys, 0.0); yl = y[:, k]
        out.append(y)
    return out
compare('Second form, fit period 1991-2020: 500 panels', Y[:, TR], Y0, simulate2(Y0, 0, int(TR.sum())))
compare('Second form, held-out years 2021-2024: 500 panels simulated forward from the 2020 record', Y[:, TE], Y[:, i20], simulate2(Y[:, i20], i20 + 1, T))

# ------------------------------------------------------------------ third form: a hidden regime
# Both forms above are driven by the recorded deaths alone. The third form adds a state the record does
# not show: each country is in one of K hidden regimes, moves between them with fixed probabilities,
# and in each regime records zero with some probability and otherwise a log death count drawn around
# the regime's level. With K = 3 this is the hidden Markov model already ranked among the estimators
# (code/state_estimators.py). It is fitted on 1991-2020 by the Baum-Welch algorithm (maximum
# likelihood) and judged on the same twelve statistics. A regime that records mostly zeros and often
# moves to a violent regime is a state that builds before deaths are recorded.
def hmm_fit(Yseq, K, n_iter=200):
    if K == 3:
        A = np.array([[.97, .025, .005], [.15, .75, .10], [.03, .17, .80]]); pi = np.array([.8, .15, .05])
        p0 = np.array([.97, .25, .02]); m = np.array([1.5, 3.5, 7.0]); s = np.array([1.0, 1.2, 1.2])
    else:
        A = np.array([[.95, .04, .008, .002], [.20, .70, .08, .02], [.05, .10, .75, .10], [.02, .03, .15, .80]]); pi = np.array([.7, .15, .1, .05])
        p0 = np.array([.99, .80, .20, .02]); m = np.array([1.2, 2.0, 4.0, 7.0]); s = np.array([0.8, 1.0, 1.2, 1.2])
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
    return A, pi, p0, m, s, np.log(sc).sum(), al
def hmm_sim(pars, s0, steps, n=500):
    A, pi, p0, m, s = pars[:5]; K = len(pi); cumA = np.cumsum(A, 1); out = []
    for _ in range(n):
        st = s0(); y = np.zeros((C, steps))
        for k in range(steps):
            st = (rng.random(C)[:, None] > cumA[st]).sum(1).clip(0, K - 1)
            pos = rng.random(C) >= p0[st]
            v = m[st] + s[st] * rng.normal(size=C)
            y[:, k] = np.where(pos & (v >= C0), v, 0.0)
        out.append(y)
    return out
Ytr = np.concatenate([Y0[:, None], Y[:, TR]], 1)                  # 1990-2020
for K in (3, 4):
    pars = hmm_fit(Ytr, K); A, pi, p0, m, s, ll, al = pars
    print(f'\nthird form, {K} hidden regimes, fitted on 1990-2020: log-likelihood {ll:.0f}')
    for k in range(K):
        print(f'  regime {k + 1}: records zero in {100 * p0[k]:.0f} percent of years; level when recorded exp({m[k]:.1f}) = {np.exp(m[k]) - 1:.0f} deaths; '
              f'stays {100 * A[k, k]:.0f} percent; moves to ' + ', '.join(f'{j + 1}: {100 * A[k, j]:.1f}' for j in range(K) if j != k))
    post = lambda t: (lambda: (rng.random(C)[:, None] > np.cumsum(al[:, t], 1)).sum(1).clip(0, K - 1))
    compare(f'Third form ({K} regimes), fit period 1991-2020: 500 panels', Y[:, TR], Y0, hmm_sim(pars, post(0), int(TR.sum())))
    compare(f'Third form ({K} regimes), held-out years 2021-2024: 500 panels simulated forward from 2020', Y[:, TE], Y[:, i20], hmm_sim(pars, post(al.shape[1] - 1), int(TE.sum())))

# ------------------------------------------------------------------ fourth form: hidden tension plus persistence in war
# The second form reproduces the course of wars and misses their silent beginnings; the hidden-regime
# form does the reverse. The fourth form joins them. Each country carries a hidden two-valued state,
# calm or tense, that moves with fixed probabilities and is never observed. In a year that follows no
# recorded deaths, the level of the latent violence depends on that hidden state; in a year that follows
# recorded deaths, violence carries over at rate rho, as in the second form. Fitted on 1991-2020 by
# maximum likelihood with the forward algorithm.
def parts4(th):
    sg = lambda v: 1 / (1 + np.exp(-v))
    return dict(b=th[0:4], calm=th[4], tense=th[4] + np.exp(th[5]), A=th[6], rho=sg(th[7]), sq=np.exp(th[8]), sa=np.exp(th[9]), pct=sg(th[10]), ptc=sg(th[11]))
def emis4(P, y, ylag, t):
    cov = Xs[:, t, :] @ P['b']; act = ylag > 0
    out = np.zeros((C, 2))
    for k, lev in enumerate([P['calm'], P['tense']]):
        mq = cov + lev; ma = cov + P['A'] + P['rho'] * ylag
        m = np.where(act, ma, mq); sd = np.where(act, P['sa'], P['sq'])
        out[:, k] = np.where(y > 0, norm.pdf(y, m, sd), norm.cdf(C0, m, sd))
    return out + 1e-300
def fwd4(th, upto):
    P = parts4(th); Tm = np.array([[1 - P['pct'], P['pct']], [P['ptc'], 1 - P['ptc']]])
    st = np.array([P['ptc'], P['pct']]) / (P['pct'] + P['ptc']); al = np.tile(st, (C, 1)); ll = 0.0
    for t in range(upto):
        al = (al @ Tm) * emis4(P, Y[:, t], YL[:, t], t); sc = al.sum(1); ll += np.log(sc).sum(); al /= sc[:, None]
    return ll, al
ntr = int(TR.sum()); best4 = None
for start in [[-.5, .5, -.3, -.3, -4, 1.0, 1.0, 1.0, 0.3, 0.2, -3, -1], [0, 0, 0, 0, -6, 1.5, 1.5, 0.5, 0.5, 0.5, -4, 0], [-.5, .5, -.3, -.3, -3, 0.5, 0.5, 1.5, 0.0, 0.0, -2, -2]]:
    r4 = minimize(lambda th: -fwd4(th, ntr)[0], np.array(start, float), method='L-BFGS-B')
    if best4 is None or r4.fun < best4.fun: best4 = r4
th4 = best4.x; P4 = parts4(th4)
print(f"\nfourth form, fitted on 1991-2020 by maximum likelihood: log-likelihood {-best4.fun:.0f}; persistence in active years rho {P4['rho']:.2f}; "
      f"quiet-year level calm {P4['calm']:.2f}, tense {P4['tense']:.2f}; yearly chance calm to tense {100 * P4['pct']:.1f} percent, tense to calm {100 * P4['ptc']:.1f} percent; "
      f"share of time tense {100 * P4['pct'] / (P4['pct'] + P4['ptc']):.0f} percent")
def sim4(al0, ystart, t0, t1, n=500):
    Tm = np.array([[1 - P4['pct'], P4['pct']], [P4['ptc'], 1 - P4['ptc']]]); out = []
    for _ in range(n):
        s = (rng.random(C) < al0[:, 1]).astype(int); yl = ystart.copy(); y = np.zeros((C, t1 - t0))
        for k, t in enumerate(range(t0, t1)):
            s = (rng.random(C) < Tm[s, 1]).astype(int)
            cov = Xs[:, t, :] @ P4['b']; act = yl > 0
            m = np.where(act, cov + P4['A'] + P4['rho'] * yl, cov + np.where(s == 1, P4['tense'], P4['calm']))
            ys = m + np.where(act, P4['sa'], P4['sq']) * rng.normal(size=C)
            y[:, k] = np.where(ys >= C0, ys, 0.0); yl = y[:, k]
        out.append(y)
    return out
st4 = np.array([P4['ptc'], P4['pct']]) / (P4['pct'] + P4['ptc'])
compare('Fourth form, fit period 1991-2020: 500 panels', Y[:, TR], Y0, sim4(np.tile(st4, (C, 1)), Y0, 0, ntr))
compare('Fourth form, held-out years 2021-2024: 500 panels simulated forward from 2020', Y[:, TE], Y[:, i20], sim4(fwd4(th4, ntr)[1], Y[:, i20], i20 + 1, T))
