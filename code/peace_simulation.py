"""Simulation of the reduced peace model: behavior over time on each side of the stability conditions.

The model is the one in peace_game.py: the discrete-time form of the violence, fear, and reserve
equations and the Richardson arms pair, for two neighboring states, with the sensor equation
(observation late, weighted by transparency chi, rumor filling the rest) and institutions that
set enforcement in proportion to observed violence. All coefficients are ILLUSTRATIVE; none is
estimated from data. The simulation tests the model against its own stability conditions. It is
not evidence about the record.

Three experiments:
 1. Behavior over time. Four pairs of runs, one on each side of a condition: trust (revenge loop),
    transparency (fear loop), reaction to the rival's arms (arms race), and total delay.
    Each run is 500 plays with random grievance shocks; the median and the 10-90 percent band are saved.
 2. The trust threshold. Failure share against trust, with two analytic thresholds: the trust
    below which the enforcement ceiling cannot damp violence, and the trust below which the
    enforcement needed costs more than the reserve takes in.
 3. Sensitivity. 3,000 parameter sets are drawn, every coefficient between half and one and a
    half times its base value, with transparency, trust, delay, institutional gain, and arms
    reaction drawn over their ranges. For each set the four single-loop conditions are checked
    on paper, and 40 plays are simulated. The question: do conditions derived one loop at a
    time tell whether the coupled model fails? A fifth quantity is then checked: the reserve
    balance at the level where the coupled model settles (function operating_point).

Run from the package root:  python code/peace_simulation.py     (about two minutes; no data needed)
Writes outputs/peace_simulation_output.txt (when redirected) and outputs/peace_simulation_series.npz.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

BASE = dict(b1=1.0, b2=0.25, b3=0.60, b4=0.05, T=0.60, c1=0.30, c2h=0.20, c3=0.05, c5=0.40, c6=0.10, rho=0.60, kappa=1.0, cu=0.50,
            lam=0.15, m=0.30, R0=12.0, umax=3.0, sig=0.05, Vwar=8.0, H=60, pshock=0.12, shock_lo=0.4, shock_hi=1.2)
RULE = dict(K=1.5, tau_a=1, k_arm=0.10)

def simulate(p, rule, chi, tau_d, n, seed):
    """n plays at once. Arrays are plays x states(2) x rounds. Same equations as peace_game.play."""
    rng = np.random.default_rng(seed); H = p['H']
    V = np.zeros((n, 2, H + 1)); F = np.zeros((n, 2, H + 1)); A = np.zeros((n, 2, H + 1)); R = np.zeros((n, 2, H + 1)); R[:, :, 0] = p['R0']
    Y = np.zeros((n, 2, H)); U = np.zeros((n, 2, H)); alive = np.ones(n, bool); fail_round = np.full(n, H + 1)
    G = np.where(rng.random((n, 2, H)) < p['pshock'], rng.uniform(p['shock_lo'], p['shock_hi'], (n, 2, H)), 0.0)
    for t in range(H):
        td = max(t - tau_d, 0); ta = t - rule['tau_a']
        Y[:, :, t] = np.maximum(chi * V[:, :, td] + (1 - chi) * F[:, :, t] + p['sig'] * rng.standard_normal((n, 2)), 0.0)
        want = rule['K'] * Y[:, :, ta] if ta >= 0 else np.zeros((n, 2))
        threat = chi * A[:, ::-1, td] + (1 - chi) * F[:, :, t]
        u = np.minimum(want, p['umax']); ua = rule['k_arm'] * threat
        cost = p['cu'] * u + p['kappa'] * ua
        s = np.where(cost > R[:, :, t], np.maximum(R[:, :, t], 0.0) / np.where(cost > 0, cost, 1.0), 1.0)
        u = u * s; ua = ua * s; U[:, :, t] = u
        Vn = np.maximum(V[:, :, t] + p['b1'] * G[:, :, t] + p['b2'] * V[:, :, t] - (p['b3'] * u * p['T'] + p['b4'] * p['T']) * V[:, :, t], 0.0)
        Fn = np.maximum(F[:, :, t] + p['c1'] * V[:, :, t] + p['c2h'] * F[:, :, t] + p['c3'] * A[:, ::-1, t] - (p['c5'] * chi + p['c6']) * F[:, :, t], 0.0)
        An = np.maximum(A[:, :, t] - p['m'] * A[:, :, t] + ua, 0.0)
        Rn = np.maximum(R[:, :, t] + p['rho'] - p['cu'] * u - p['kappa'] * ua - p['lam'] * F[:, :, t], 0.0)
        # a play that has failed is frozen at its last values
        V[:, :, t + 1] = np.where(alive[:, None], Vn, V[:, :, t]); F[:, :, t + 1] = np.where(alive[:, None], Fn, F[:, :, t])
        A[:, :, t + 1] = np.where(alive[:, None], An, A[:, :, t]); R[:, :, t + 1] = np.where(alive[:, None], Rn, R[:, :, t])
        newfail = alive & (V[:, :, t + 1] > p['Vwar']).any(1); fail_round[newfail] = t + 1; alive &= ~newfail
    return dict(V=V, F=F, A=A, R=R, U=U, fail=~alive, fail_round=fail_round)

def conditions(p, rule, chi, delay):
    """The four single-loop conditions, checked on paper, and the budget condition."""
    arms = rule['k_arm'] < p['m']                                   # restraint outweighs reaction
    fear = p['c2h'] < p['c6'] + p['c5'] * chi                       # hoarding gain below the dissipation of fear
    need = (p['b2'] - p['b4'] * p['T']) / (p['b3'] * p['T'])        # enforcement that holds violence level
    revenge = need < p['umax']                                      # b2 < b3*u*T + b4*T with enforcement at its ceiling
    dly = delay < 1.0 / max(p['b2'], p['c2h'])                      # correction arrives before the disturbance spreads
    budget = p['cu'] * need < p['rho']                              # the enforcement needed costs less than the reserve takes in
    return arms, fear, revenge, dly, budget

def operating_point(p, rule, chi):
    """The level at which the coupled model settles when institutions respond in proportion to
    observed violence, and the reserve balance there. Enforcement settles at the value that holds
    violence level, need = (b2 - b4*T)/(b3*T). Fear settles in proportion to violence, arms in
    proportion to fear. The balance is what the reserve takes in each round minus the cost of that
    enforcement, the reserve drawn by hoarding, and the upkeep of arms. A negative balance empties
    the reserve, after which enforcement cannot be paid for."""
    need = (p['b2'] - p['b4'] * p['T']) / (p['b3'] * p['T']); D = p['c5'] * chi + p['c6'] - p['c2h']; den = p['m'] - rule['k_arm'] * chi
    if D <= 0 or den <= 0: return need, np.inf, np.inf, np.inf, -np.inf
    V = need / (rule['K'] * (chi + (1 - chi) * p['c1'] / D)); F = p['c1'] * V / D
    for _ in range(50):                                             # arms and fear feed each other; iterate to the fixed point
        A = rule['k_arm'] * (1 - chi) * F / den; Fn = (p['c1'] * V + p['c3'] * A) / D
        if abs(Fn - F) < 1e-10: break
        F = Fn
    return need, V, F, A, p['rho'] - p['cu'] * need - p['lam'] * F - p['kappa'] * p['m'] * A

if __name__ == '__main__':
    q = lambda x: np.percentile(x, [10, 50, 90], axis=0)
    # ------------------------------------------------------------ 1. behavior over time
    print('1. Behavior over time: 500 plays on each side of each condition; failure share and the median at rounds 10, 30, 60')
    series = {}
    def pair(tag, var, lab, kw_ok, kw_bad):
        for side, kw in (('stable', kw_ok), ('unstable', kw_bad)):
            p = dict(BASE, **kw.get('p', {})); rule = dict(RULE, **kw.get('rule', {})); chi = kw.get('chi', 0.9); td = kw.get('tau_d', 1)
            o = simulate(p, rule, chi, td, 500, 1); x = o[var][:, 0, :]; series[f'{tag}_{side}'] = q(x); series[f'{tag}_{side}_fail'] = o['fail'].mean()
            c = conditions(p, rule, chi, td + rule['tau_a']); med = np.median(x, 0)
            print(f'  {lab:22s} {side:8s} {str(kw):58s} failure share {o["fail"].mean():.2f}; median {var} at rounds 10/30/60: {med[10]:.2f} {med[30]:.2f} {med[60]:.2f}; conditions arms/fear/revenge/delay/budget: ' + ''.join('Y' if v else 'n' for v in c))
    pair('trust', 'V', 'trust (revenge loop)', dict(p=dict(T=0.6)), dict(p=dict(T=0.2)))
    pair('transp', 'F', 'transparency (fear)', dict(chi=0.9), dict(chi=0.2))
    pair('arms', 'A', 'arms reaction', dict(rule=dict(k_arm=0.10)), dict(rule=dict(k_arm=0.45)))
    pair('delay', 'V', 'total delay', dict(rule=dict(tau_a=1), tau_d=1), dict(rule=dict(tau_a=5), tau_d=3))
    # ------------------------------------------------------------ 2. the trust threshold
    b = BASE; T_ceiling = b['b2'] / (b['b3'] * b['umax'] + b['b4']); T_budget = b['b2'] / (b['b3'] * b['rho'] / b['cu'] + b['b4'])
    print(f'\n2. Failure share against trust (transparency 0.9, 1,000 plays each)')
    print(f'   analytic thresholds: enforcement ceiling T > {T_ceiling:.3f}; budget T > {T_budget:.3f}')
    Ts = np.round(np.arange(0.05, 0.95, 0.05), 2); fs = []; rr = []
    for T in Ts:
        o = simulate(dict(BASE, T=T), RULE, 0.9, 1, 1000, 2); fs.append(o['fail'].mean()); rr.append(np.median(o['R'][:, 0, -1]))
        print(f'   trust {T:.2f}: failure share {fs[-1]:.2f}; median reserve at the end {rr[-1]:.1f}')
    series['trust_grid'] = Ts; series['trust_fail'] = np.array(fs); series['T_ceiling'] = T_ceiling; series['T_budget'] = T_budget
    # ------------------------------------------------------------ 3. sensitivity over parameter sets
    print('\n3. Sensitivity: 3,000 parameter sets, 40 plays each')
    rng = np.random.default_rng(7); rows = []
    for d in range(3000):
        p = dict(BASE)
        for k in ['b1', 'b2', 'b3', 'b4', 'c1', 'c2h', 'c3', 'c5', 'c6', 'rho', 'cu', 'lam', 'm']: p[k] = BASE[k] * rng.uniform(0.5, 1.5)
        p['T'] = rng.uniform(0.1, 0.9); chi = rng.uniform(0.1, 1.0); tau_d = int(rng.integers(1, 4)); tau_a = int(rng.integers(1, 6))
        rule = dict(K=rng.uniform(0.5, 3.0), tau_a=tau_a, k_arm=rng.uniform(0.05, 0.5))
        c = conditions(p, rule, chi, tau_d + tau_a); f = simulate(p, rule, chi, tau_d, 40, 100 + d)['fail'].mean()
        bal = operating_point(p, rule, chi); rows.append(c + (f, bal[4], bal[1]))
    Rw = np.array(rows, float); C4 = Rw[:, :4]; bud = Rw[:, 4]; f = Rw[:, 5]; nv = 4 - C4.sum(1)
    print(f'   mean failure share over all sets {f.mean():.3f}; sets that fail in most plays {np.mean(f > 0.5):.3f}')
    print('   by number of the four conditions violated:   sets   mean failure share   share of sets failing in most plays')
    by = []
    for k in range(5):
        m_ = nv == k
        if m_.any(): print(f'     {k} violated: {int(m_.sum()):5d}   {f[m_].mean():.3f}   {np.mean(f[m_] > 0.5):.3f}'); by.append((k, int(m_.sum()), f[m_].mean()))
    series['sens_by'] = np.array(by)
    ok4 = nv == 0
    print(f'   all four met: {int(ok4.sum())} sets, mean failure share {f[ok4].mean():.3f}; with the budget condition also met: {int((ok4 & (bud == 1)).sum())} sets, {f[ok4 & (bud == 1)].mean():.3f}; four met and budget violated: {int((ok4 & (bud == 0)).sum())} sets, {f[ok4 & (bud == 0)].mean():.3f}')
    names = ['arms race', 'fear loop', 'revenge loop', 'delay', 'budget']
    print('   each condition alone: mean failure share when met / when violated, and share of sets where it is violated')
    for j, nme in enumerate(names): print(f'     {nme:13s} {f[Rw[:, j] == 1].mean():.3f} / {f[Rw[:, j] == 0].mean():.3f}   violated in {np.mean(Rw[:, j] == 0):.2f}')
    y = (f > 0.5).astype(int)
    print(f'   AUC for "fails in most plays": number of the four violated {roc_auc_score(y, nv):.3f}; with the budget condition as a fifth {roc_auc_score(y, nv + (1 - bud)):.3f}')
    h = len(y) // 2
    for lab, X in (('four conditions', C4), ('five conditions', Rw[:, :5])):
        mdl = LogisticRegression().fit(1 - X[:h], y[:h]); print(f'   logistic on the {lab}, fitted on half the sets and scored on the other half: AUC {roc_auc_score(y[h:], mdl.predict_proba(1 - X[h:])[:, 1]):.3f}; weights ' + ' '.join(f'{w:+.2f}' for w in mdl.coef_[0]))
    bal = Rw[:, 6]; Vs = Rw[:, 7]; ok = np.isfinite(bal) & (bal > 0) & (Vs < BASE['Vwar'])
    print('\n   the reserve balance at the operating point of the coupled model (one inequality in place of four)')
    print(f'   balance positive: {int(ok.sum())} sets, mean failure share {f[ok].mean():.3f}; balance negative or no operating point: {int((~ok).sum())} sets, {f[~ok].mean():.3f}')
    print(f'   AUC of the balance for "fails in most plays": {roc_auc_score(y, -np.clip(np.nan_to_num(bal, neginf=-99.0), -99, 99)):.3f}')
    five = Rw[:, :5].sum(1) == 5
    print(f'   five conditions met and balance positive: {int((five & ok).sum())} sets, {f[five & ok].mean():.3f}; five met and balance negative: {int((five & ~ok).sum())} sets, {f[five & ~ok].mean():.3f}')
    edges = [0, 0.1, 0.2, 0.3, 0.5, 9]; mb = []
    print('   failure share by size of the positive balance (base replenishment is 0.6 a round):')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m_ = ok & (bal > lo) & (bal <= hi); mb.append((lo, hi, int(m_.sum()), f[m_].mean())); print(f'     {lo:.1f} to {hi:.1f}: {int(m_.sum()):4d} sets   {f[m_].mean():.3f}')
    series['balance_bins'] = np.array(mb); series['balance_neg'] = f[~ok].mean()
    np.savez('outputs/peace_simulation_series.npz', **series)
