"""
Which measurements would identify which condition? A test on synthetic data with known truth.

The fear, reserve, and arms equations of the model cannot be fitted, because no cross-country series
exists for fear, enforcement effort, or the arms pair. This script asks what each missing series
would buy if it were collected. It simulates the reduced model of code/peace_simulation.py with its
base coefficients, so the true value of every coefficient is known, and then estimates the
coefficients by least squares from what an analyst would hold under four sets of measurements:

  A. the public record only: the observed violence y (late, weighted by transparency, rumor filling the rest)
  B. the record and a series of enforcement effort u (police and court activity)
  C. the record and a series of fear F (a repeated survey of fear or hoarding)
  D. the two arms series A (military spending of each side) and the fear series

Each case is run at three levels of transparency. The data are synthetic and the coefficients
illustrative: the result shows what the estimator can recover from each set of series. It is not
evidence about any country.

Run from the package root: python code/identifiability_test.py   (about one minute; no data needed)
"""
import sys, numpy as np
sys.path.insert(0, 'code')
from peace_simulation import simulate, BASE, RULE
p, rule = dict(BASE), dict(RULE); TD = 1
def ols(y, X):
    X = np.column_stack(X); return np.linalg.lstsq(X, y, rcond=None)[0]
def one(chi, seed, n=100):
    o = simulate(p, rule, chi, TD, n, seed); H = p['H']; rng = np.random.default_rng(seed + 10 ** 6)
    V, F, A, U = o['V'], o['F'], o['A'], o['U']
    Yv = np.zeros((n, 2, H))
    for t in range(H):                                    # the analyst's violence series: the sensor of the model
        Yv[:, :, t] = np.maximum(chi * V[:, :, max(t - TD, 0)] + (1 - chi) * F[:, :, t] + p['sig'] * rng.standard_normal((n, 2)), 0)
    ok = (np.arange(H)[None, :] < (o['fail_round'][:, None] - 2))            # rounds before a play fails
    m = np.repeat(ok[:, None, :], 2, 1)[:, :, 1:H - 1]
    y0, y1 = Yv[:, :, 1:H - 1][m], Yv[:, :, 2:H][m]; u0 = U[:, :, 1:H - 1][m]
    f0, f1 = F[:, :, 1:H - 1][m], F[:, :, 2:H][m]
    a0, a1 = A[:, :, 1:H - 1][m], A[:, :, 2:H][m]; ar_lag = A[:, ::-1, 0:H - 2][m]; ar0 = A[:, ::-1, 1:H - 1][m]
    one_ = np.ones_like(y0); r = {}
    r['A: persistence of violence from the record alone'] = ols(y1 - y0, [one_, y0])[1]
    cB = ols(y1 - y0, [one_, y0, u0 * y0]); r['B: open-loop persistence, with an enforcement series'] = cB[1]; r['B: enforcement-by-trust gain b3*T'] = -cB[2]
    cC = ols(f1 - f0, [one_, y0, f0, ar0]); r['C: net gain of the fear loop, with a fear series'] = cC[2]; r['C: violence-to-fear gain c1'] = cC[1]
    cD = ols(a1 - a0, [one_, a0, ar_lag, f0]); r['D: restraint m, with both arms series'] = -cD[1]; r['D: reaction to the rival (k times transparency)'] = cD[2]
    return r
print('Truth and recovered values: median and 10-90 percent range over 100 synthetic panels of 100 plays each')
for chi in (0.9, 0.6, 0.3):
    truth = {'A: persistence of violence from the record alone': p['b2'] - p['b4'] * p['T'],
             'B: open-loop persistence, with an enforcement series': p['b2'] - p['b4'] * p['T'], 'B: enforcement-by-trust gain b3*T': p['b3'] * p['T'],
             'C: net gain of the fear loop, with a fear series': p['c2h'] - p['c5'] * chi - p['c6'], 'C: violence-to-fear gain c1': p['c1'],
             'D: restraint m, with both arms series': p['m'], 'D: reaction to the rival (k times transparency)': rule['k_arm'] * chi}
    R = [one(chi, s) for s in range(100)]
    fails = np.mean([simulate(p, rule, chi, TD, 100, s)['fail'].mean() for s in range(5)])
    print(f'\ntransparency {chi}   (share of plays that fail: {100 * fails:.0f} percent)')
    print(f"{'quantity':58s} {'truth':>7s} {'median':>8s} {'10-90 percent':>18s}  sign right")
    for k in truth:
        v = np.array([x[k] for x in R]); lo, md, hi = np.percentile(v, [10, 50, 90])
        print(f'{k:58s} {truth[k]:7.2f} {md:8.2f} {lo:8.2f} to {hi:5.2f}   {100 * np.mean(np.sign(v) == np.sign(truth[k])):3.0f} percent')
