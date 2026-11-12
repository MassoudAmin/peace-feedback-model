"""
The three fitted equations joined as one coupled system.

State of a country in year t:  x[t] = (v, mil, res)
   v    log(1 + battle deaths) (UCDP);  mil  military spending as a share of GDP (SIPRI);
   res  total reserves in months of imports (World Bank).
Coupled form:    x[t+1] = c_i + A x[t] + e[t]      A is 3 by 3, c_i a constant vector for each country.
Uncoupled form:  the same with the six off-diagonal entries of A set to zero.

Fitted by least squares with country constants on 1991-2020; intervals from 500 resamples of
countries. Reported: the nine entries of A, the eigenvalues of A (all inside the unit circle means the
fitted system returns to its country mean after a shock), the gain around the loop
violence -> arms -> reserves -> violence, a joint test of the six cross terms by permutation, and the
held-out error for 2021-2024 one year ahead and as a free run from the 2020 state.
Reads panels/panel2.csv and panels/panel4.csv. Run from the package root.
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(20261112)
V = ['ldeaths', 'mil', 'res_mo']; L = ['violence', 'arms', 'reserves']
p = pd.read_csv('panels/panel2.csv')[['country', 'year', 'mil', 'ldeaths']]
r = pd.read_csv('panels/panel4.csv')[['country', 'year', 'res_mo']]
d = p.merge(r, on=['country', 'year']).dropna().sort_values(['country', 'year'])
nx = d.copy(); nx['year'] -= 1
d = d.merge(nx.rename(columns={v: v + '1' for v in V}), on=['country', 'year'])
V1 = [v + '1' for v in V]
tr = d[d.year <= 2019]

def fit(df, coupled=True):
    g = df.groupby('country'); X = (df[V] - g[V].transform('mean')).values; Y = (df[V1] - g[V1].transform('mean')).values
    if coupled: return np.linalg.lstsq(X, Y, rcond=None)[0].T
    return np.diag([np.linalg.lstsq(X[:, [j]], Y[:, j], rcond=None)[0][0] for j in range(3)])
def resample(df):
    cs = df.country.unique(); pick = rng.choice(cs, len(cs))
    return pd.concat([df[df.country == c].assign(country=f'{c}_{i}') for i, c in enumerate(pick)])

A = fit(tr); D = fit(tr, False); cs = tr.country.unique()
B = np.array([fit(resample(tr)) for _ in range(500)])
lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
print(f'== Coupled system: {len(tr)} country-years, {len(cs)} countries, fitted on 1991-2020')
print('   entry of A (row: next year; column: this year)')
for i in range(3):
    for j in range(3):
        print(f'   {L[i]:9s} <- {L[j]:9s} {A[i, j]:+.3f}  (95 percent interval {lo[i, j]:+.3f} to {hi[i, j]:+.3f})')
ev = np.linalg.eigvals(A); rho = np.abs(ev).max(); rb = np.array([np.abs(np.linalg.eigvals(b)).max() for b in B])
print(f'   eigenvalues of A: {np.round(np.sort(np.abs(ev))[::-1], 3)}; largest in size {rho:.3f} (interval {np.percentile(rb, 2.5):.3f} to {np.percentile(rb, 97.5):.3f}); below 1 in {100 * np.mean(rb < 1):.0f} percent of resamples')
print(f'   largest eigenvalue of the uncoupled form: {np.abs(np.diag(D)).max():.3f}')
print(f'   half-life of the slowest mode: {np.log(0.5) / np.log(rho):.1f} years')
loop = A[1, 0] * A[2, 1] * A[0, 2]; lb = B[:, 1, 0] * B[:, 2, 1] * B[:, 0, 2]
print(f'   loop gain violence -> arms -> reserves -> violence: {loop:+.5f} (interval {np.percentile(lb, 2.5):+.5f} to {np.percentile(lb, 97.5):+.5f})')

# joint test of the six cross terms: shuffle which country's cross-variable histories go with which own history
def sse(df, coupled):
    g = df.groupby('country'); X = (df[V] - g[V].transform('mean')).values; Y = (df[V1] - g[V1].transform('mean')).values
    M = fit(df, coupled); return ((Y - X @ M.T) ** 2).sum(axis=0)
gain = 1 - sse(tr, True) / sse(tr, False)
print('   share of in-sample error removed by the cross terms: ' + ', '.join(f'{l} {100 * g:.2f} percent' for l, g in zip(L, gain)))
null = []
for _ in range(500):
    s = tr.copy(); g = s.groupby('country'); X = (s[V] - g[V].transform('mean')).values; Y = (s[V1] - g[V1].transform('mean')).values
    out = []
    for j in range(3):
        o = [k for k in range(3) if k != j]; Xp = X.copy(); Xp[:, o] = X[rng.permutation(len(X))][:, o]
        e1 = ((Y[:, j] - Xp @ np.linalg.lstsq(Xp, Y[:, j], rcond=None)[0]) ** 2).sum()
        e0 = ((Y[:, j] - X[:, [j]] @ np.linalg.lstsq(X[:, [j]], Y[:, j], rcond=None)[0]) ** 2).sum(); out.append(1 - e1 / e0)
    null.append(out)
null = np.array(null)
print('   chance of a share that large with unrelated cross series: ' + ', '.join(f'{l} {max(np.mean(null[:, j] >= gain[j]), 1 / 500):.3f}' for j, l in enumerate(L)))

# held-out years
mu = tr.groupby('country')[V].mean()
te = d[(d.year >= 2020) & (d.year <= 2023) & d.country.isin(cs)]
X = te[V].values - mu.loc[te.country].values; Y = te[V1].values - mu.loc[te.country].values
print(f'\n== Held-out, one year ahead, 2021-2024: {len(te)} country-years')
for name, M in [('coupled', A), ('uncoupled', D)]:
    e = ((Y - X @ M.T) ** 2).mean(axis=0); print(f'   {name:10s} squared error: ' + ', '.join(f'{l} {x:.3f}' for l, x in zip(L, e)))
e = ((Y - X) ** 2).mean(axis=0); print('   no change  squared error: ' + ', '.join(f'{l} {x:.3f}' for l, x in zip(L, e)))
# free run from the 2020 state
full = d[d.country.isin(cs)]; s0 = full[full.year == 2020].set_index('country')
res = {n: [] for n in ['coupled', 'uncoupled', 'no change']}; n = 0
for c in s0.index:
    path = full[(full.country == c) & (full.year >= 2020) & (full.year <= 2023)].sort_values('year')
    if len(path) < 4: continue
    n += 1; x0 = s0.loc[c, V].values.astype(float) - mu.loc[c].values; truth = path[V1].values - mu.loc[c].values
    for name, M in [('coupled', A), ('uncoupled', D), ('no change', np.eye(3))]:
        x = x0.copy(); sim = []
        for _ in range(4): x = M @ x; sim.append(x.copy())
        res[name].append((np.array(sim) - truth) ** 2)
print(f'\n== Held-out, free run from the 2020 state through 2024: {n} countries')
for name in res:
    e = np.mean(res[name], axis=(0, 1)); print(f'   {name:10s} squared error: ' + ', '.join(f'{l} {x:.3f}' for l, x in zip(L, e)))

# bracket the largest eigenvalue: country constants bias persistence down in short panels, and no constants bias it up
Xp = tr[V].values - tr[V].values.mean(axis=0); Yp = tr[V1].values - tr[V1].values.mean(axis=0); Ap = np.linalg.lstsq(Xp, Yp, rcond=None)[0].T
long = tr[tr.groupby('country').country.transform('size') >= 25]; Al = fit(long)
print(f'\n== Bounds on the largest eigenvalue')
print(f'   with country constants {rho:.3f}; without them {np.abs(np.linalg.eigvals(Ap)).max():.3f}; with constants, countries with 25 or more years ({long.country.nunique()} countries) {np.abs(np.linalg.eigvals(Al)).max():.3f}')
print(f'   couplings into violence without constants: arms {Ap[0, 1]:+.3f}, reserves {Ap[0, 2]:+.3f}; into reserves: violence {Ap[2, 0]:+.3f}, arms {Ap[2, 1]:+.3f}')

# does the coupling into violence depend on the regime? refit the violence row on country-years with fighting under way
act = tr[tr.ldeaths > 0]; act = act[act.groupby('country').country.transform('size') >= 5]
def row(df):
    g = df.groupby('country'); X = (df[V] - g[V].transform('mean')).values; y = (df['ldeaths1'] - g['ldeaths1'].transform('mean')).values
    return np.linalg.lstsq(X, y, rcond=None)[0]
a = row(act); ab = np.array([row(resample(act)) for _ in range(500)]); l2, h2 = np.percentile(ab, [2.5, 97.5], axis=0)
print(f'\n== Violence row refitted on years with fighting under way: {len(act)} country-years, {act.country.nunique()} countries')
for j in range(3): print(f'   violence  <- {L[j]:9s} {a[j]:+.3f}  (95 percent interval {l2[j]:+.3f} to {h2[j]:+.3f})')

# Is violence a first-order process? Compare the measured coefficient at each lag with the one-month coefficient compounded.
# Uses the monthly panel, which the package does not ship (it carries ACLED columns); monthly_panel.py rebuilds it.
import os
if os.path.exists('panels/monthly_panel.csv'):
    m = pd.read_csv('panels/monthly_panel.csv')[['country', 'ym', 'sb']].sort_values(['country', 'ym'])
    m = m[(m.groupby('country').sb.transform('sum') > 0) & (m.ym < '2025')].copy(); m['z'] = np.log1p(m.sb)
    print(f'\n== Persistence of monthly log(1 + deaths) by lag, {m.country.nunique()} countries with recorded deaths, 1989-2024')
    print('   lag in months   pooled   around each country\'s mean   one-month coefficient compounded')
    for k in [1, 3, 6, 12, 24]:
        m['zk'] = m.groupby('country').z.shift(-k); b = m.dropna(subset=['zk']); g = b.groupby('country')
        zd = b.z - g.z.transform('mean'); kd = b.zk - g.zk.transform('mean'); pooled = np.polyfit(b.z, b.zk, 1)[0]
        if k == 1: p1 = pooled
        print(f'   {k:13d}   {pooled:6.2f}   {(zd @ kd) / (zd @ zd):26.2f}   {p1 ** k:31.2f}')
