"""
Two more equations of the model on public series: the arms equation and the reserve equation.

Arms (Richardson form, one side):   mil[t+1] - mil[t] = g_i - m mil[t] + k nbr[t] + c v[t]
   mil  military spending as a share of GDP (SIPRI); nbr  mean of the same across land neighbors;
   v    log(1 + battle deaths) at home; g_i a constant for each country.
   m is restraint, k is reaction to the neighbors. With one m and one k for all countries, the
   arms-race condition m m' > k k' reduces to m > k.
Reserves:   res[t+1] - res[t] = r_i - a res[t] - kappa mil[t] - theta v[t]
   res  total reserves in months of imports (World Bank). The model says arms spending and violence
   draw the reserve down (kappa > 0, theta > 0).

Both are fitted by least squares with country constants on 1991-2020; intervals come from 500
resamples of countries. The held-out check predicts the change in 2021-2024 with the fitted
coefficients and compares the squared error with a forecast of no change beyond each country's mean.
Reads panels/panel2.csv, panels/panel4.csv, panels/borders.csv. Run from the package root.
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(20261112)
p = pd.read_csv('panels/panel2.csv')[['country', 'year', 'mil', 'ldeaths']]
r = pd.read_csv('panels/panel4.csv')[['country', 'year', 'res_mo']]
b = pd.read_csv('panels/borders.csv')
m = p.merge(b, on='country').merge(p.rename(columns={'country': 'neighbour', 'mil': 'nmil'})[['neighbour', 'year', 'nmil']], on=['neighbour', 'year'])
nbr = m.groupby(['country', 'year']).nmil.mean().rename('nbr').reset_index()
d = p.merge(nbr, on=['country', 'year'], how='left').merge(r, on=['country', 'year'], how='left').sort_values(['country', 'year'])
nx = d[['country', 'year', 'mil', 'res_mo']].copy(); nx['year'] -= 1
d = d.merge(nx.rename(columns={'mil': 'mil1', 'res_mo': 'res1'}), on=['country', 'year'], how='left')
d['dmil'] = d.mil1 - d.mil; d['dres'] = d.res1 - d.res_mo

def fit(df, y, xs):
    df = df.dropna(subset=[y] + xs); g = df.groupby('country')
    Yd = df[y] - g[y].transform('mean'); Xd = df[xs] - g[xs].transform('mean')
    return np.linalg.lstsq(Xd.values, Yd.values, rcond=None)[0], df
def run(title, y, xs, labels):
    tr = d[d.year <= 2019]; te = d[(d.year >= 2020) & (d.year <= 2023)]      # the change runs from year to year + 1
    beta, used = fit(tr, y, xs); cs = used.country.unique(); boots = []
    for _ in range(500):
        pick = rng.choice(cs, len(cs)); parts = [used[used.country == c].assign(country=f'{c}_{i}') for i, c in enumerate(pick)]
        boots.append(fit(pd.concat(parts), y, xs)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print(f'\n== {title}: {len(used)} country-years, {len(cs)} countries, fitted on changes 1991-2020')
    for l, bb, a, c in zip(labels, beta, lo, hi): print(f'   {l:46s} {bb:+.3f}   (95 percent interval {a:+.3f} to {c:+.3f})')
    te = te.dropna(subset=[y] + xs); te = te[te.country.isin(cs)]
    mu = used.groupby('country')[[y] + xs].mean()
    pred = mu.loc[te.country, y].values + ((te[xs].values - mu.loc[te.country, xs].values) @ beta)
    e1 = np.mean((te[y].values - pred) ** 2); e0 = np.mean((te[y].values - mu.loc[te.country, y].values) ** 2)
    print(f'   held-out changes 2021-2024: {len(te)} country-years; squared error {e1:.3f} against {e0:.3f} for a forecast of no change beyond the country mean ({100 * (1 - e1 / e0):+.0f} percent)')
    return beta, np.array(boots)
ba, Ba = run('Arms equation', 'dmil', ['mil', 'nbr', 'ldeaths'], ['own spending (minus restraint m)', 'neighbors\' spending (reaction k)', 'violence at home'])
mm, kk = -Ba[:, 0], Ba[:, 1]
print(f'   restraint m {-ba[0]:.3f}, reaction k {ba[1]:.3f}; m exceeds k in {100 * np.mean(mm > kk):.0f} percent of resamples; reaction k above zero in {100 * np.mean(kk > 0):.0f} percent')
run('Reserve equation', 'dres', ['res_mo', 'mil', 'ldeaths'], ['own reserves (minus adjustment a)', 'military spending (minus kappa)', 'violence at home (minus theta)'])
