"""
Ranks the state estimators of state_estimators.py over four rolling test windows, so that the
ranking rests on more onsets than the eleven of the last window.

Each window fits on the country-years through its last fitting year and tests on the next years:
    fit through 2005, test 2006-2010;  fit through 2010, test 2011-2015;
    fit through 2015, test 2016-2020;  fit through 2020, test 2021-2024.
Run from the package root: python code/estimator_ranking.py
"""
import os, sys, json, subprocess, tempfile
import numpy as np
WIN = [(2005, 2006, 2010), (2010, 2011, 2015), (2015, 2016, 2020), (2020, 2021, 2024)]
res = []
for tr, lo, hi in WIN:
    f = tempfile.mktemp(suffix='.json')
    env = dict(os.environ, TR_END=str(tr), TE_LO=str(lo), TE_HI=str(hi), FAST='1', DUMP=f)
    subprocess.run([sys.executable, 'code/state_estimators.py'], env=env, check=True, stdout=subprocess.DEVNULL)
    res.append({r['name']: r for r in json.load(open(f))})
names = list(res[0].keys())
rows = []
for n in names:
    a = [r[n]['auc'] for r in res]; w = [r[n]['war'] for r in res]
    rows.append((n, np.mean(a), a, sum(r[n]['top'] for r in res), sum(r[n]['n'] for r in res), np.mean(w), sum(r[n]['wtop'] for r in res), sum(r[n]['wn'] for r in res)))
rows.sort(key=lambda r: -r[1])
print('Onsets per window:', [res[k][names[0]]['n'] for k in range(4)], ' escalations to war per window:', [res[k][names[0]]['wn'] for k in range(4)])
print(f"{'rank':4s} {'estimator':58s} {'mean onset AUC':>14s}  {'2006-10':>7s} {'2011-15':>7s} {'2016-20':>7s} {'2021-24':>7s}  {'top tenth':>10s}  {'mean war AUC':>12s} {'top tenth':>10s}")
for k, r in enumerate(rows):
    print(f"{k+1:<4d} {r[0]:58s} {r[1]:14.3f}  " + ' '.join(f'{x:7.3f}' for x in r[2]) + f"  {f'{r[3]} of {r[4]}':>10s}  {r[5]:12.3f} {f'{r[6]} of {r[7]}':>10s}")
best = rows[0][1]
print(f'\nEstimators within 0.02 of the best mean onset AUC ({best:.3f}): ' + '; '.join(r[0] for r in rows if best - r[1] <= 0.02))
