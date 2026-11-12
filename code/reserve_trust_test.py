"""Does the record show the pattern the simulation predicts? (Chapter 11; Appendix D)

peace_simulation.py finds that the coupled model fails through its reserve: failure comes where
the enforcement that low trust requires costs more than the reserve takes in. The coefficients of
that simulation are illustrative, so its numbers cannot be compared with the record. Its pattern
can: new violence should be most frequent where the capacity to contain violence is low AND the
reserve is thin, more than where only one of the two is true.

Three tests, for each kind of organized violence that UCDP records (state-based; non-state:
communal, sectarian, militia; one-sided: killing of civilians) and for any kind:
 1. Trust and reserves. The 62 countries of World Values Survey wave 7, predictor years 2020-2024,
    onsets 2021-2025, by interpersonal trust (median split) and reserves (three months of imports).
 2. Rule of law and reserves. Every country-year with a reserves figure, 1991-2024: onset rate by
    cell; the interaction in a logistic regression with a country bootstrap; and the out-of-sample
    gain from adding reserves and the interaction, over four test windows.
 3. Regions. 768 survey regions with regional trust and regional deaths in 2020-2025: is the link
    between regional trust and regional deaths steeper in countries with thin reserves?

Reserves are total reserves in months of imports (World Bank FI.RES.TOTL.MO), the one reserve series
with wide coverage. It is a weak proxy for the food, fuel, credit, and fiscal room of the model.

Run from the package root, after code/monthly_panel.py:  python code/reserve_trust_test.py
"""
import warnings, numpy as np, pandas as pd
from scipy.stats import fisher_exact, beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); rng = np.random.default_rng(0)

m = pd.read_csv('panels/monthly_panel.csv', usecols=['country', 'ym', 'sb', 'ns', 'os']); m['year'] = m.ym.str[:4].astype(int); m = m[m.year <= 2025]
yr = m.groupby(['country', 'year'])[['sb', 'ns', 'os']].sum().reset_index(); yr['any'] = yr.sb + yr.ns + yr['os']
KINDS = {'sb': 'state-based', 'ns': 'non-state (communal, sectarian, militia)', 'os': 'one-sided (killing of civilians)', 'any': 'any kind'}
yr = yr.sort_values(['country', 'year'])
for k in KINDS:
    g = yr.groupby('country')[k]; nxt = g.shift(-1); prev = g.shift(1)
    yr[k + '_risk'] = (yr[k] < 25) & (prev < 25); yr[k + '_on'] = np.where(nxt.isna(), np.nan, (nxt >= 25).astype(float))
p = pd.read_csv('panels/panel4.csv')[['country', 'year', 'res_mo', 'lres', 'v2x_rule', 'since46', 'ever46', 'mil']]
d = yr.merge(p, on=['country', 'year'], how='inner')
def exact(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k); return lo, hi
def cells(df, a, b, la, lb, kind):
    r = df[df[kind + '_risk'] & df[kind + '_on'].notna()]; out = []
    for va, na in ((False, f'{la} low'), (True, f'{la} high')):
        for vb, nb in ((False, f'{lb} low'), (True, f'{lb} high')):
            s = r[(r[a] == va) & (r[b] == vb)]; k_ = int(s[kind + '_on'].sum()); n_ = len(s); lo, hi = exact(k_, n_) if n_ else (np.nan, np.nan)
            out.append((na, nb, k_, n_)); print(f'     {na:18s} {nb:14s} {k_:3d} of {n_:4d}  {100 * k_ / max(n_, 1):5.1f}%  ({100 * lo:.1f} to {100 * hi:.1f})')
    ll = out[0]; rest = (sum(o[2] for o in out[1:]), sum(o[3] for o in out[1:]))
    if ll[3] and rest[1]:
        orr, pv = fisher_exact([[ll[2], ll[3] - ll[2]], [rest[0], rest[1] - rest[0]]])
        print(f'     both low against the other three cells: {100 * ll[2] / ll[3]:.1f}% against {100 * rest[0] / rest[1]:.1f}%; Fisher exact p = {pv:.3f}')
    # is it more than either alone? compare both-low with the two one-low cells
    one = (out[1][2] + out[2][2], out[1][3] + out[2][3])
    if one[1]: print(f'     both low against the two cells with only one low: {100 * ll[2] / max(ll[3], 1):.1f}% against {100 * one[0] / one[1]:.1f}%; Fisher exact p = {fisher_exact([[ll[2], ll[3] - ll[2]], [one[0], one[1] - one[0]]])[1]:.3f}')

# ------------------------------------------------------------ 1. trust and reserves, surveyed countries
w = pd.read_csv('panels/wvs7_country_measures_weighted.csv')[['country', 'trust']]
t1 = d[(d.year >= 2020) & (d.year <= 2024)].merge(w, on='country'); med = t1.drop_duplicates('country').trust.median()
t1['trust_hi'] = t1.trust >= med; t1['res_hi'] = t1.res_mo >= 3
print(f'1. Trust and reserves: {t1.country.nunique()} surveyed countries with a reserves figure, predictor years 2020-2024; trust split at its median ({med:.1f} percent), reserves at three months of imports')
for k, lab in KINDS.items():
    print(f'   {lab}: onsets / country-years at risk, rate (exact 95% interval)'); cells(t1, 'trust_hi', 'res_hi', 'trust', 'reserves', k)

# ------------------------------------------------------------ 2. rule of law and reserves, all countries with reserves
d['rule_hi'] = d.v2x_rule >= 0.5; d['res_hi'] = d.res_mo >= 3; d2 = d[d.v2x_rule.notna() & d.since46.notna()].copy()
print(f'\n2. Rule of law and reserves: {d2.country.nunique()} countries, {len(d2)} country-years, 1991-2024; rule of law split at 0.5, reserves at three months of imports')
d2['h1'] = np.log1p(d2.since46); d2['h2'] = np.log1p(d2.ever46); d2['x'] = d2.v2x_rule * d2.lres
BASEV = ['h1', 'h2', 'v2x_rule']; FULL = BASEV + ['lres', 'x']
def fit(df, cols, y):
    sc = StandardScaler().fit(df[cols]); return sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(df[cols]), y)
for k, lab in KINDS.items():
    print(f'   {lab}: onsets / country-years at risk, rate (exact 95% interval)'); cells(d2, 'rule_hi', 'res_hi', 'rule of law', 'reserves', k)
    r = d2[d2[k + '_risk'] & d2[k + '_on'].notna()]; y = r[k + '_on'].values
    sc, mdl = fit(r, FULL, y); co = dict(zip(FULL, mdl.coef_[0])); cs = r.country.unique(); B = []
    for _ in range(500):
        s = pd.concat([r[r.country == c] for c in rng.choice(cs, len(cs))])
        if s[k + '_on'].sum() < 5: continue
        B.append(fit(s, FULL, s[k + '_on'].values)[1].coef_[0])
    B = np.array(B); q = np.percentile(B, [2.5, 97.5], axis=0)
    print(f'     logistic, standardized weights (95% interval, countries resampled): log reserves {co["lres"]:+.2f} ({q[0][3]:+.2f} to {q[1][3]:+.2f}); rule of law x log reserves {co["x"]:+.2f} ({q[0][4]:+.2f} to {q[1][4]:+.2f}); {int(y.sum())} onsets')
    g = []; gb = []
    for (e, lo, hi) in [(2005, 2006, 2010), (2010, 2011, 2015), (2015, 2016, 2020), (2020, 2021, 2024)]:
        tr = r[r.year <= e - 1]; te = r[(r.year >= lo - 1) & (r.year <= hi - 1)]
        if te[k + '_on'].sum() < 1 or tr[k + '_on'].sum() < 5: continue
        s0, m0 = fit(tr, BASEV, tr[k + '_on'].values); s1, m1 = fit(tr, FULL, tr[k + '_on'].values)
        p0 = m0.predict_proba(s0.transform(te[BASEV]))[:, 1]; p1 = m1.predict_proba(s1.transform(te[FULL]))[:, 1]; yt = te[k + '_on'].values
        g.append(roc_auc_score(yt, p1) - roc_auc_score(yt, p0)); gb.append((te.country.values, yt, p0, p1))
    bs = []
    for _ in range(500):
        v = []
        for (cc, yt, p0, p1) in gb:
            u = np.unique(cc); pick = rng.choice(u, len(u)); idx = np.concatenate([np.where(cc == c)[0] for c in pick])
            if 0 < yt[idx].sum() < len(idx): v.append(roc_auc_score(yt[idx], p1[idx]) - roc_auc_score(yt[idx], p0[idx]))
        if v: bs.append(np.mean(v))
    print(f'     out of sample, four windows: adding reserves and the interaction to history and rule of law changes the mean AUC by {np.mean(g):+.3f} ({np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}); windows with a gain {int((np.array(g) > 0).sum())} of {len(g)}')

# ------------------------------------------------------------ 3. regions
rg = pd.read_csv('panels/wvs7_regions.csv'); rs = p[(p.year >= 2018) & (p.year <= 2020)].groupby('country').res_mo.mean().rename('res'); rg = rg.merge(rs, on='country'); rg = rg[rg.n >= 30]
act = rg.groupby('country').deaths.transform('sum') >= 100; rg = rg[act]; rg['thin'] = rg.res < 3
print(f'\n3. Regions: {len(rg)} survey regions in {rg.country.nunique()} countries with 100 or more deaths in 2020-2025 and a reserves figure; {rg[rg.thin].country.nunique()} countries below three months of imports')
def slope(df):
    x = df.trust - df.groupby('country').trust.transform('mean'); yv = df.ld - df.groupby('country').ld.transform('mean'); return float((x * yv).sum() / (x * x).sum())
def boot(df, f, n=1000):
    cs = df.country.unique(); out = []
    for _ in range(n):
        s = pd.concat([df[df.country == c].assign(country=f'{c}_{i}') for i, c in enumerate(rng.choice(cs, len(cs)))]); out.append(f(s))
    return np.nanpercentile(out, [2.5, 97.5])
for lab, sub in (('reserves under three months', rg[rg.thin]), ('reserves of three months or more', rg[~rg.thin])):
    if sub.country.nunique() >= 3:
        lo, hi = boot(sub, slope); print(f'   {lab}: {len(sub)} regions in {sub.country.nunique()} countries; within-country slope of log deaths on trust (per percentage point) {slope(sub):+.3f} ({lo:+.3f} to {hi:+.3f})')
    else: print(f'   {lab}: {len(sub)} regions in {sub.country.nunique()} countries; too few countries for a slope')
if rg[rg.thin].country.nunique() >= 3:
    diff = lambda s: slope(s[s.thin]) - slope(s[~s.thin]) if s.thin.any() and (~s.thin).any() else np.nan
    lo, hi = boot(rg, diff); print(f'   difference in slope, thin minus ample reserves: {diff(rg):+.3f} ({lo:+.3f} to {hi:+.3f})')

# ------------------------------------------------------------ 4. fiscal room in place of foreign-exchange reserves
# World Bank Cross-Country Database of Fiscal Space (Kose, Kurlat, Ohnsorge, and Sugawara 2022), Fall 2025 version.
# Four measures, fixed before the test was run: government debt and the fiscal balance as shares of GDP, government
# debt as a share of average tax revenues, and the sovereign rating. Each is turned so that a higher value means more room.
fs = pd.read_csv('panels/fiscal_space.csv'); MEAS = {'ggdy': ('government debt, % of GDP', -1), 'fby': ('fiscal balance, % of GDP', +1), 'dfggd': ('government debt, % of average tax revenues', -1), 'sovrate': ('sovereign rating', +1)}
d4 = yr.merge(p[['country', 'year', 'v2x_rule', 'since46', 'ever46']], on=['country', 'year']).merge(fs, on=['country', 'year'], how='left')
d4 = d4[d4.v2x_rule.notna() & d4.since46.notna()].copy(); d4['rule_hi'] = d4.v2x_rule >= 0.5; d4['h1'] = np.log1p(d4.since46); d4['h2'] = np.log1p(d4.ever46)
print('\n4. Rule of law and fiscal room, 1991-2024; thin fiscal room is the worse half of each measure; sixteen comparisons, so about one would reach p < 0.05 by chance')
for v, (lab, sgn) in MEAS.items():
    s = d4[d4[v].notna()].copy(); z = sgn * s[v]; z = z.clip(z.quantile(0.01), z.quantile(0.99)); s['z'] = (z - z.mean()) / z.std(); s['res_hi'] = s.z >= s.z.median(); s['x'] = s.v2x_rule * s.z
    print(f'  {lab}: {s.country.nunique()} countries, {len(s)} country-years')
    for k, klab in KINDS.items():
        r = s[s[k + '_risk'] & s[k + '_on'].notna()]; lowr = r[~r.rule_hi]; a = lowr[~lowr.res_hi]; b = lowr[lowr.res_hi]; hi_ = r[r.rule_hi]
        ka, na, kb, nb = int(a[k + '_on'].sum()), len(a), int(b[k + '_on'].sum()), len(b); pv = fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
        y = r[k + '_on'].values; cols = ['h1', 'h2', 'v2x_rule', 'z', 'x']; co = fit(r, cols, y)[1].coef_[0]; cs_ = r.country.unique(); B = []
        for _ in range(200):
            q = pd.concat([r[r.country == c] for c in rng.choice(cs_, len(cs_))])
            if q[k + '_on'].sum() >= 5: B.append(fit(q, cols, q[k + '_on'].values)[1].coef_[0])
        B = np.array(B); lo3, hi3 = np.percentile(B[:, 3], [2.5, 97.5]); lo4, hi4 = np.percentile(B[:, 4], [2.5, 97.5]); g = []
        for (e, lo, hi) in [(2005, 2006, 2010), (2010, 2011, 2015), (2015, 2016, 2020), (2020, 2021, 2024)]:
            tr = r[r.year <= e - 1]; te = r[(r.year >= lo - 1) & (r.year <= hi - 1)]
            if te[k + '_on'].sum() < 1 or tr[k + '_on'].sum() < 5: continue
            s0, m0 = fit(tr, cols[:3], tr[k + '_on'].values); s1, m1 = fit(tr, cols, tr[k + '_on'].values); yt = te[k + '_on'].values
            g.append(roc_auc_score(yt, m1.predict_proba(s1.transform(te[cols]))[:, 1]) - roc_auc_score(yt, m0.predict_proba(s0.transform(te[cols[:3]]))[:, 1]))
        print(f'    {klab:42s} low rule of law: thin room {ka} of {na} ({100 * ka / max(na, 1):.1f}%), ample room {kb} of {nb} ({100 * kb / max(nb, 1):.1f}%), Fisher p {pv:.3f}; high rule of law {int(hi_[k + "_on"].sum())} of {len(hi_)}; '
              f'weight on room {co[3]:+.2f} ({lo3:+.2f} to {hi3:+.2f}), on rule of law x room {co[4]:+.2f} ({lo4:+.2f} to {hi4:+.2f}); out-of-sample AUC change {np.mean(g):+.3f}, gain in {int((np.array(g) > 0).sum())} of {len(g)} windows')
# trust and fiscal room, surveyed countries
t4 = d4[(d4.year >= 2020) & (d4.year <= 2024)].merge(w, on='country'); t4['trust_hi'] = t4.trust >= t4.drop_duplicates('country').trust.median()
print('  trust and fiscal room, surveyed countries, predictor years 2020-2024, onsets of any kind: low trust with thin room / low trust with ample room / high trust')
for v, (lab, sgn) in MEAS.items():
    s = t4[t4[v].notna() & t4['any_risk'] & t4['any_on'].notna()].copy(); z = sgn * s[v]; s['ample'] = z >= z.median(); lt = s[~s.trust_hi]
    a = lt[~lt.ample]; b = lt[lt.ample]; h_ = s[s.trust_hi]
    print(f'    {lab:44s} {int(a.any_on.sum())} of {len(a)} / {int(b.any_on.sum())} of {len(b)} / {int(h_.any_on.sum())} of {len(h_)}   ({s.country.nunique()} countries)')
