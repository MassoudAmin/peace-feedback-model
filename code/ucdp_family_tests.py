"""Four further tests of the peace model's terms on the UCDP family of datasets.

A. Repair: peace agreements (UCDP PA 22.2) and whether fighting in the dyad continues.
B. Coupling: external support (UCDP ESD 18.1) and the persistence/escalation of dyad violence.
C. Delay: third-party management events (UCDP MIC 22.2) — how long after a dyad first fights does
   outside correction arrive, and does fighting continue after it.
D. Legitimacy disturbance: deadly electoral violence (DECO 1.0) and next-year onset/escalation in the
   country-year panel of the replication package.
All results are conditional frequencies on dyad-years or country-years; none is a causal estimate.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
D = sys.argv[1] if len(sys.argv) > 1 else 'data'       # directory with the UCDP files
PKGROOT = sys.argv[2] if len(sys.argv) > 2 else '.'       # replication package root (panels/panel2.csv)
import os; OUT = os.path.join(PKGROOT, 'outputs'); os.makedirs(OUT, exist_ok=True)
from scipy.stats import beta as _beta
def wilson(k, n):
    if n == 0: return (np.nan, np.nan)
    z = 1.96; p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0, c-h), min(1, c+h))
def fmt(k, n):
    lo, hi = wilson(k, n); return f'{k}/{n} = {100*k/n:.1f}% [{100*lo:.0f}–{100*hi:.0f}]' if n else 'n/a'

bd = pd.read_excel(f'{D}/BattleDeaths_v26_1.xlsx')
dy = pd.read_excel(f'{D}/Dyadic_v26_1.xlsx')
# dyad-year deaths 1989–2025 (bd_best); active = >= 25
deaths = bd.groupby(['dyad_id','year']).bd_best.sum()
years = range(1989, 2026)
dyads = sorted(set(bd.dyad_id))
M = pd.DataFrame(0, index=dyads, columns=list(years))
for (d, y), v in deaths.items(): M.loc[d, y] = v
def active(d, y): return y in M.columns and d in M.index and M.loc[d, y] >= 25
def dead(d, y): return M.loc[d, y] if (d in M.index and y in M.columns) else 0
first_active = dy[dy.year >= 1946].groupby('dyad_id').year.min()

out = []
def P(s=''): print(s); out.append(s)

# ---------------- A. peace agreements ----------------
P('=== A. Peace agreements (UCDP PA 22.2, 1975–2021) and continuation of fighting in the dyad')
pa = pd.read_excel(f'{D}/ucdp-peace-agreements-222.xlsx')
pa = pa[pa.year >= 1989]
pa['dyad_id'] = pd.to_numeric(pa.dyad_id, errors='coerce')
# agreements can list several dyads separated by commas; keep single-dyad rows plus split multi
rows = []
for _, r in pa.iterrows():
    ids = str(r['dyad_id']) if pd.notna(r['dyad_id']) else ''
    for tok in str(r.get('dyad_id', '')).replace(';', ',').split(','):
        try: rows.append((int(float(tok)), int(r.year), r))
        except: pass
P(f'agreement rows 1989–2021: {len(pa)}; dyad-agreement pairs resolvable: {len(rows)}; matched to a UCDP dyad with recorded deaths: {sum(1 for d,_,_ in rows if d in M.index)}')
agr = {}
for d, y, r in rows:
    if d in M.index: agr.setdefault((d, y), []).append(r)
# active dyad-years 1989–2020 (so that t+1..t+5 exists)
rec = []
for d in dyads:
    for y in range(1989, 2021):
        if not active(d, y): continue
        has = (d, y) in agr
        nxt = active(d, y+1)
        any5 = any(active(d, y+k) for k in range(1, 6))
        ratio = np.log((dead(d, y+1) + 1) / (dead(d, y) + 1))
        feats = {}
        if has:
            rs = agr[(d, y)]
            def f1(x, k):
                v = pd.to_numeric(x.get(k, 0), errors='coerce'); return (not pd.isna(v)) and int(v) == 1
            feats = dict(third=any(pd.notna(x.get('pa_3rd')) and str(x.get('pa_3rd')).strip() != '' for x in rs), full=any(int(x['pa_type']) == 1 for x in rs if pd.notna(x['pa_type'])), ddr=any(f1(x, 'ddr') for x in rs),
                         pp=any(f1(x, 'pp') for x in rs), cease=any(f1(x, 'cease') for x in rs), ended=any(f1(x, 'ended') for x in rs))
        rec.append(dict(dyad=d, year=y, agr=has, nxt=nxt, any5=any5, ratio=ratio, **feats))
R = pd.DataFrame(rec)
a, b = R[R.agr], R[~R.agr]
P(f'active dyad-years 1989–2020: {len(R)}; with an agreement signed that year: {len(a)}')
P(f'  still active next year:        with agreement {fmt(int(a.nxt.sum()), len(a))}   without {fmt(int(b.nxt.sum()), len(b))}')
P(f'  active in any of next 5 years: with agreement {fmt(int(a.any5.sum()), len(a))}   without {fmt(int(b.any5.sum()), len(b))}')
P(f'  median log(deaths t+1 / deaths t): with agreement {a.ratio.median():+.2f}   without {b.ratio.median():+.2f}')
for col, lab in [('third', 'third party in the agreement'), ('full', 'full (comprehensive) agreement'), ('ddr', 'DDR provision'), ('pp', 'power-sharing provision'), ('cease', 'ceasefire provision')]:
    x, y_ = a[a[col] == True], a[a[col] == False]
    P(f'  {lab:40s}: next-year active  yes {fmt(int(x.nxt.sum()), len(x))}  no {fmt(int(y_.nxt.sum()), len(y_))};  5-year  yes {fmt(int(x.any5.sum()), len(x))}  no {fmt(int(y_.any5.sum()), len(y_))}')
# agreement fate per PA dataset
pa_all = pd.read_excel(f'{D}/ucdp-peace-agreements-222.xlsx')
P(f'  of all {len(pa_all)} agreements 1975–2021, UCDP codes {int((pa_all.ended == True).sum())} as ended')
R.to_csv(f'{OUT}/out_A_agreements.csv', index=False)

# ---------------- B. external support ----------------
P('\n=== B. External support (UCDP ESD 18.1, dyad-year, 1975–2017) and persistence of dyad violence')
es = pd.read_excel(f'{D}/ucdp-esd-dy-181.xlsx')
tr = pd.read_csv(f'{D}/translate_dyad.csv') if False else None
es = es[(es.year >= 1989) & (es.year <= 2016)]
# ESD 18.1 uses old dyad ids where available; map through translate_dyad.csv (new_id, old_id)
import os
tfile = f'{D}/translate_dyad.csv'
tdy = pd.read_csv(tfile); old2new = {}
for _, r in tdy.iterrows():
    try: old2new[str(r.old_id)] = int(r.new_id)
    except: pass
def newid(x):
    s = str(x).strip()
    if s in old2new: return old2new[s]
    try:
        v = int(float(s)); return v if v in M.index else (old2new.get(str(v)) or v)
    except: return None
es['dyad_new'] = es.dyad_id
es = es.dropna(subset=['dyad_new']); es['dyad_new'] = es.dyad_new.astype(int)
P(f'ESD dyad-years 1989–2016: {len(es)}; matched to dyads with recorded deaths: {int(es.dyad_new.isin(M.index).sum())}')
g = es.groupby(['dyad_new', 'year']).agg(sup=('ext_sup', 'max'), both=('ext_bothsides', 'max')).reset_index()
rows = []
for _, r in g.iterrows():
    d, y = int(r.dyad_new), int(r.year)
    if d not in M.index or not active(d, y): continue
    rows.append(dict(dyad=d, year=y, sup=int(0 if pd.isna(r.sup) else r.sup), both=int(0 if pd.isna(r.both) else r.both), nxt=active(d, y+1),
                     esc=(dead(d, y) < 1000 and dead(d, y+1) >= 1000), low=dead(d, y) < 1000,
                     ratio=np.log((dead(d, y+1)+1)/(dead(d, y)+1))))
E = pd.DataFrame(rows)
P(f'active dyad-years in ESD coverage: {len(E)}')
for lab, sub in [('no external support recorded', E[E.sup == 0]), ('support to one side', E[(E.sup == 1) & (E.both == 0)]), ('support to both sides', E[E.both == 1])]:
    lo = sub[sub.low]
    P(f'  {lab:32s}: active next year {fmt(int(sub.nxt.sum()), len(sub))};  escalation to >=1,000 next year (from below) {fmt(int(lo.esc.sum()), len(lo))};  median log ratio {sub.ratio.median():+.2f}')
E.to_csv(f'{OUT}/out_B_support.csv', index=False)

# ---------------- C. third-party management ----------------
P('\n=== C. Third-party management (UCDP MIC 22.2, 1993–2007): delay of outside correction, and fighting after it')
mic = pd.read_csv(f'{D}/micFINAL.csv', low_memory=False)
mic['date'] = pd.to_datetime(mic.start_event, errors='coerce'); mic = mic.dropna(subset=['date'])
mic['year'] = mic.date.dt.year
confd = bd.groupby(['conflict_id','year']).bd_best.sum()
CM = confd.unstack(fill_value=0).reindex(columns=list(years), fill_value=0)
def cactive(c, y): return y in CM.columns and c in CM.index and CM.loc[c, y] >= 25
def cdead(c, y): return CM.loc[c, y] if (c in CM.index and y in CM.columns) else 0
cfirst = dy.groupby('conflict_id').year.min()
mic = mic.dropna(subset=['conflict_id'])
tc = pd.read_csv(f'{D}/translate_conf.csv')
o2n = {}
for _, r in tc.iterrows():
    try: o2n[int(float(str(r.old_id)))] = int(r.new_id)
    except: pass
mic['conflict_id'] = [o2n.get(int(c), int(c)) for c in mic.conflict_id]
P(f'management events: {len(mic)}; matched conflicts: {int(mic.conflict_id.isin(CM.index).sum())} events in {mic[mic.conflict_id.isin(CM.index)].conflict_id.nunique()} conflicts')
lags = []
for c, grp in mic.groupby('conflict_id'):
    if c not in cfirst.index: continue
    fa = int(cfirst[c])
    if fa < 1993 or fa > 2005: continue
    lags.append(int(grp.year.min()) - fa)
lags = np.array(lags)
if len(lags):
    P(f'conflicts first active 1993–2005 with any management event: {len(lags)}; lag from first active year to first event: median {np.median(lags):.0f} years, mean {lags.mean():.1f}, share within the first year {np.mean(lags<=0):.2f}, within two years {np.mean(lags<=1):.2f}')
covered = set(mic.conflict_id)
P(f'MIC covers intrastate conflicts in Africa only: {len(covered)} conflicts; comparisons below are within those conflicts')
med = mic[mic.type_of_talks.isin([1, 2, 3])]
mset = set(zip(med.conflict_id, med.year)); aset = set(zip(mic.conflict_id, mic.year))
rows = []
for c in covered:
    for y in range(1993, 2007):
        if c not in CM.index or not cactive(c, y): continue
        rows.append(dict(conflict=c, year=y, any_ev=(c, y) in aset, talks=(c, y) in mset, nxt=cactive(c, y+1), ratio=np.log((cdead(c, y+1)+1)/(cdead(c, y)+1))))
C = pd.DataFrame(rows)
P(f'active conflict-years 1993–2006: {len(C)}')
for lab, sub in [('no management event that year', C[~C.any_ev]), ('any management event', C[C.any_ev]), ('talks held', C[C.talks])]:
    P(f'  {lab:34s}: active next year {fmt(int(sub.nxt.sum()), len(sub))};  median log ratio {sub.ratio.median():+.2f}')
C.to_csv(f'{OUT}/out_C_management.csv', index=False)

# ---------------- D. electoral violence ----------------
P('\n=== D. Deadly electoral violence (DECO 1.0, 1989–2017) and next-year onset in the country-year panel')
de = pd.read_excel(f'{D}/DECO_fixed.xlsx' if os.path.exists(f'{D}/DECO_fixed.xlsx') else f'{D}/DECO_v.1.0.xlsx')  # see data/README.md on the DECO workbook
de = de[(de.electoral_vio == 1) & (de.best > 0)]
cy = de.groupby(['country', 'year']).best.sum()
pan = pd.read_csv(f'{PKGROOT}/panels/panel2.csv')
pan = pan[(pan.year >= 1991) & (pan.year <= 2017)].copy()
pan['ev'] = [cy.get((c, y), 0) for c, y in zip(pan.country, pan.year)]
pan['ev_prev'] = [cy.get((c, y-1), 0) for c, y in zip(pan.country, pan.year)]
matched = set(de.country) & set(pan.country)
P(f'DECO countries with deadly electoral violence: {de.country.nunique()}; matched to panel names: {len(matched)}; unmatched: {sorted(set(de.country)-set(pan.country))[:12]}')
for target in ['onset', 'onset1000']:
    x = pan.dropna(subset=[target])
    a, b = x[x.ev_prev > 0], x[x.ev_prev == 0]
    P(f'  {target:10s} next year | deadly electoral violence in the previous year: {fmt(int(a[target].sum()), len(a))}   | none: {fmt(int(b[target].sum()), len(b))}')
    a2 = x[x.ev_prev >= 25]
    P(f'  {target:10s} next year | >=25 electoral deaths previous year: {fmt(int(a2[target].sum()), len(a2))}')
pan.to_csv(f'{OUT}/out_D_electoral.csv', index=False)
open(f'{OUT}/ucdp_family_tests_output.txt', 'w').write('\n'.join(out))
