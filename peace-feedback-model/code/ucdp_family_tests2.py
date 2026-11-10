"""Three more checks on the UCDP family (second batch).

E. How conflicts end and whether they come back: UCDP Conflict Termination Dataset v4-2024, episodes
   ending 1989–2019, recurrence of the same conflict within five years by type of termination.
F. The panel's onset coding against UCDP's own onset dataset (25.1), 1991–2020.
G. What the fight is about: UCDP Issues Dataset 23.2 (dyad-year, to 2017) and persistence of dyad violence.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
D = sys.argv[1] if len(sys.argv) > 1 else 'data'
PKGROOT = sys.argv[2] if len(sys.argv) > 2 else '.'
import os; OUT = os.path.join(PKGROOT, 'outputs'); os.makedirs(OUT, exist_ok=True)
def wilson(k, n):
    if n == 0: return (np.nan, np.nan)
    z = 1.96; p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0, c-h), min(1, c+h))
def fmt(k, n):
    lo, hi = wilson(k, n); return f'{k}/{n} = {100*k/n:.1f}% [{100*lo:.0f}–{100*hi:.0f}]' if n else 'n/a'
out = []
def P(s=''): print(s); out.append(s)

# ---------------- E. termination and recurrence ----------------
P('=== E. Conflict termination (UCDP Conflict Termination v4-2024) and recurrence within five years')
t = pd.read_csv(f'{D}/UCDPConflictTerminationDataset_v4_2024_Conflict.csv')
ep = t.groupby('c_epid').agg(conflict=('conflict_id', 'first'), start=('c_ep_startyear', 'first'), end=('c_ep_endyear', 'first'),
                             outcome=('c_outcome', 'first'), typ=('type_of_conflict', 'first')).reset_index()
ep = ep.dropna(subset=['end']); ep['end'] = ep.end.astype(int)
starts = ep.groupby('conflict').start.apply(sorted).to_dict()
def recur(row, k=5):
    return any(row.end < s <= row.end + k for s in starts.get(row.conflict, []))
E = ep[(ep.end >= 1989) & (ep.end <= 2019)].copy()
E['recur5'] = [recur(r) for _, r in E.iterrows()]
E['recur2'] = [recur(r, 2) for _, r in E.iterrows()]
labels = {1: 'Peace agreement', 2: 'Ceasefire agreement', 3: 'Victory for the government', 4: 'Victory for the rebels', 5: 'Low activity (fighting fell below 25 deaths)', 6: 'Actor ceased to exist'}
P(f'conflict episodes ending 1989–2019: {len(E)}; recurrence of the same conflict within five years: {fmt(int(E.recur5.sum()), len(E))}; within two years: {fmt(int(E.recur2.sum()), len(E))}')
rows_E = []
for code, lab in labels.items():
    s = E[E.outcome == code]
    P(f'  {lab:48s}: episodes {len(s):3d}; recurred within 5 years {fmt(int(s.recur5.sum()), len(s))}; within 2 years {fmt(int(s.recur2.sum()), len(s))}')
    rows_E.append((lab, len(s), int(s.recur5.sum())))
intra = E[E.typ.isin([3, 4])]
P(f'  intrastate only ({len(intra)} episodes): peace agreement {fmt(int(intra[intra.outcome==1].recur5.sum()), int((intra.outcome==1).sum()))}; low activity {fmt(int(intra[intra.outcome==5].recur5.sum()), int((intra.outcome==5).sum()))}; government victory {fmt(int(intra[intra.outcome==3].recur5.sum()), int((intra.outcome==3).sum()))}')
E.to_csv(f'{OUT}/out_E_termination.csv', index=False)

# ---------------- F. onset coding check ----------------
P('\n=== F. The panel’s onset coding against the UCDP onset dataset 25.1 (intrastate), 1991–2020')
o = pd.read_csv(f'{D}/ucdp-intrastate-conflict-onset-251.csv')
pan = pd.read_csv(f'{PKGROOT}/panels/panel2.csv')
pan = pan[(pan.year >= 1991) & (pan.year <= 2020)]
fx = {'DR Congo (Zaire)': 'DR Congo (Zaire)', 'Russia (Soviet Union)': 'Russia (Soviet Union)'}
o['country'] = o.name
# the panel codes onset at year t for a conflict that begins in t+1 (the row is the at-risk year); align UCDP's year t+1 with it
x = o[['country', 'year', 'onset1', 'onset2', 'newconf']].copy(); x['year'] = x.year - 1
m = pan.merge(x, on=['country', 'year'], how='left')
P(f'panel at-risk country-years 1991–2020: {int(pan.onset.notna().sum())}; matched to the onset dataset: {int(m.dropna(subset=["onset","onset2"]).shape[0])}; panel countries not in the onset dataset: {sorted(set(pan.country) - set(o.country))}')
mm = m.dropna(subset=['onset', 'onset2'])
a = mm[mm.onset == 1]; b2 = mm[mm.onset2 == 1]; b1 = mm[mm.onset1 == 1]
P(f'  panel onsets (new conflict at 25 deaths after two quiet years): {len(a)}')
P(f'  of these, coded by UCDP as onset2 (new episode after at least two inactive years) in the same year: {fmt(int((a.onset2==1).sum()), len(a))}; as onset1 (any new episode): {fmt(int((a.onset1==1).sum()), len(a))}')
P(f'  UCDP onset2 years among matched at-risk rows: {len(b2)}; of these, coded as onsets in the panel: {fmt(int((b2.onset==1).sum()), len(b2))}')
P(f'  UCDP onset1 years among matched at-risk rows: {len(b1)}; of these, coded as onsets in the panel: {fmt(int((b1.onset==1).sum()), len(b1))}')
mm.to_csv(f'{OUT}/out_F_onset_check.csv', index=False)

# ---------------- G. issues and persistence ----------------
P('\n=== G. What the fight is about (UCDP Issues Dataset 23.2) and persistence of dyad violence, 1989–2016')
bd = pd.read_excel(f'{D}/BattleDeaths_v26_1.xlsx')
deaths = bd.groupby(['dyad_id', 'year']).bd_best.sum()
years = list(range(1989, 2026)); M = deaths.unstack(fill_value=0).reindex(columns=years, fill_value=0)
def active(d, y): return d in M.index and y in M.columns and M.loc[d, y] >= 25
i = pd.read_excel(f'{D}/ucdp_issues_dataset_dyadyear_232.xlsx')
i = i[(i.year >= 1989) & (i.year <= 2016)]
i['identity'] = i.ethnicity_1.notna() | i.religion_1.notna()
i['ethnic'] = i.ethnicity_1.notna(); i['relig'] = i.religion_1.notna(); i['ideol'] = i.ideology_1.notna()
dy = pd.read_excel(f'{D}/Dyadic_v26_1.xlsx')
inc = dy.groupby('dyad_id').incompatibility.first().to_dict()
rows = []
for _, r in i.iterrows():
    d, y = int(r.dyad_id), int(r.year)
    if not active(d, y): continue
    rows.append(dict(dyad=d, year=y, identity=bool(r.identity), ethnic=bool(r.ethnic), relig=bool(r.relig), ideol=bool(r.ideol),
                     territory=(inc.get(d) == 1), nxt=active(d, y+1), nxt3=all(active(d, y+k) for k in (1, 2, 3))))
G = pd.DataFrame(rows)
P(f'active dyad-years with issue coding: {len(G)}')
for lab, sub in [('identity issue (ethnic or religious)', G[G.identity]), ('no identity issue', G[~G.identity]), ('ideological issue', G[G.ideol]),
                 ('territory incompatibility', G[G.territory]), ('government incompatibility', G[~G.territory])]:
    P(f'  {lab:40s}: active next year {fmt(int(sub.nxt.sum()), len(sub))}; active all of next three years {fmt(int(sub.nxt3.sum()), len(sub))}')
G.to_csv(f'{OUT}/out_G_issues.csv', index=False)
open(f'{OUT}/ucdp_family_tests2_output.txt', 'w').write('\n'.join(out))
