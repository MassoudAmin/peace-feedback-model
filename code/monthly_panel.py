"""Country-month panel for the monthly observer (Chapter 15, open problem 11; Appendix D).

Builds panels/monthly_panel.csv from four public sources, none of them redistributed here:
  data/GEDEvent_v26_1.csv                 UCDP Georeferenced Event Dataset 26.1, 1989-2025
  data/GEDEvent_v26_01_26_09.csv          UCDP Candidate Events, January-September 2026
  data/acled/*aggregated*.xlsx            ACLED regional weekly aggregates, to the week of October 31, 2026
  data/rtfp/WLD_RTFP_country_*.csv        World Bank Real Time Food Prices, monthly inflation by country
and from panels/panel2.csv for the yearly covariates (conflict history since 1946, rule of law,
electoral democracy), which enter each month with a one-year lag.

Columns: country, ym, sb / ns / os (UCDP deaths, state-based, non-state, one-sided), a_pv, a_fat, a_demo
(ACLED political-violence events, their reported fatalities, demonstration events; empty before a
country's first ACLED record), food (year-on-year food price inflation, percent; empty where the
series has no value), and the four covariates.

Run from the package root: python code/monthly_panel.py
"""
import glob, numpy as np, pandas as pd
idx = pd.period_range('1989-01', '2026-09', freq='M')
p = pd.read_csv('panels/panel2.csv'); countries = sorted(p.country.unique())
use = ['country', 'date_start', 'type_of_violence', 'best']
g = pd.concat([pd.read_csv('data/GEDEvent_v26_1.csv', usecols=use), pd.read_csv('data/GEDEvent_v26_01_26_09.csv', usecols=use)], ignore_index=True)
g['ym'] = pd.to_datetime(g.date_start).dt.to_period('M')
def monthly(k):
    return g[g.type_of_violence == k].groupby(['country', 'ym']).best.sum().unstack(fill_value=0).reindex(index=countries, columns=idx, fill_value=0)
sb, ns, os_ = monthly(1), monthly(2), monthly(3)

fr = [pd.read_excel(f, usecols=['WEEK', 'COUNTRY', 'EVENTS', 'FATALITIES', 'DISORDER_TYPE']) for f in sorted(glob.glob('data/acled/*aggregated*.xlsx'))]
a = pd.concat(fr, ignore_index=True); a['ym'] = pd.to_datetime(a.WEEK).dt.to_period('M')
ren = {'Democratic Republic of Congo': 'DR Congo (Zaire)', 'Republic of Congo': 'Congo', 'Russia': 'Russia (Soviet Union)',
       'Serbia': 'Serbia (Yugoslavia)', 'Yemen': 'Yemen (North Yemen)', 'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
       'Myanmar': 'Myanmar (Burma)', 'Cambodia': 'Cambodia (Kampuchea)', 'Zimbabwe': 'Zimbabwe (Rhodesia)',
       'Madagascar': 'Madagascar (Malagasy)', 'Vietnam': 'Vietnam (North Vietnam)', 'eSwatini': 'Kingdom of eSwatini (Swaziland)',
       'United States': 'United States of America', 'Samoa': 'Samoa (Western Samoa)', 'Micronesia': 'Federated States of Micronesia'}
a['country'] = a.COUNTRY.replace(ren)
pv = a.DISORDER_TYPE.str.contains('Political violence', na=False); de = a.DISORDER_TYPE.str.contains('Demonstrations', na=False)
a['pv'] = a.EVENTS * pv; a['fat'] = a.FATALITIES * pv; a['demo'] = a.EVENTS * de
start = a.groupby('country').ym.min()
def acled(col):
    m = a.groupby(['country', 'ym'])[col].sum().unstack(fill_value=0).reindex(index=countries, columns=idx).astype(float)
    for c in countries:
        if c in start.index: m.loc[c] = m.loc[c].fillna(0.0).where(idx >= start[c])      # zero inside coverage, empty before it
    return m
a_pv, a_fat, a_demo = acled('pv'), acled('fat'), acled('demo')

f = sorted(glob.glob('data/rtfp/WLD_RTFP_country_*.csv'))[-1]
r = pd.read_csv(f); r['ym'] = pd.to_datetime(r.price_date).dt.to_period('M')
r['country'] = r.country.replace({'Congo, Dem. Rep.': 'DR Congo (Zaire)', 'Congo, Rep.': 'Congo', 'Gambia, The': 'Gambia', 'Lao PDR': 'Laos',
                                  'Madagascar': 'Madagascar (Malagasy)', 'Myanmar': 'Myanmar (Burma)'})
food = r.pivot_table(index='country', columns='ym', values='Inflation').reindex(index=countries, columns=idx)

cov = ['since46', 'ever46', 'v2x_rule', 'v2x_polyarchy']
out = pd.DataFrame({'country': np.repeat(countries, len(idx)), 'ym': np.tile(idx.astype(str), len(countries))})
for nm, m in [('sb', sb), ('ns', ns), ('os', os_), ('a_pv', a_pv), ('a_fat', a_fat), ('a_demo', a_demo), ('food', food)]:
    out[nm] = m.values.reshape(-1)
yr = np.tile(idx.year.values, len(countries)) - 1                                        # one-year lag
pp = p.set_index(['country', 'year'])[cov]
full = pp.reindex(pd.MultiIndex.from_product([countries, range(1990, 2026)], names=['country', 'year'])).groupby(level=0).ffill().groupby(level=0).bfill()
out[cov] = full.reindex(list(zip(out.country, yr))).values
out.to_csv('panels/monthly_panel.csv', index=False)
print('countries', len(countries), 'months', len(idx), 'rows', len(out))
print('names not matched to the panel: ACLED', sorted(set(a.country) - set(countries)))
print('                               food', sorted(set(r.country) - set(countries)))
print('share of country-months with zero state-based deaths', round(float((out.sb == 0).mean()), 3))
print('country-months with ACLED coverage', int(out.a_pv.notna().sum()), ' with food inflation', int(out.food.notna().sum()))
print('countries with ACLED', int((a_pv.notna().sum(axis=1) > 0).sum()), ' with food', int((food.notna().sum(axis=1) > 0).sum()))
