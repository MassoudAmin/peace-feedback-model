"""The peace game: a laboratory game on the fast loops of the peace model.

Two teams each run one of two neighboring states. Every round a team sees only
the observation of equation (11.1): its own violence, late by tau_d rounds,
weighted by transparency chi, with rumor (fear) filling the rest, plus noise.
It then sets enforcement u_law and arms u_arm, both paid from a reserve.
The experimenter injects the grievance shocks and records everything.

The dynamics are the discrete-time form of equations (D.3)-(D.5) of Appendix D
and the Richardson arms pair behind (11.2), with trust held fixed for the length
of a play. All coefficients are ILLUSTRATIVE and dimensionless; none is
estimated from data. The players in this script are simulated decision rules,
not people.

Run from the package root:  python code/peace_game.py
No data files are needed. Results are reproducible from the seeds below.
"""
import numpy as np

# ---- illustrative coefficients (one round = one decision period) ------------
P = dict(
    b1=1.0,    # grievance into violence                      (D.4)
    b2=0.25,   # persistence of violence, the open-loop gain   (D.4)
    b3=0.60,   # enforcement-by-trust removal of violence      (D.4)
    b4=0.05,   # trust-only removal of violence                (D.4)
    T=0.60,    # interpersonal trust, held fixed in a play
    c1=0.30,   # violence into fear                            (D.5)
    c2h=0.20,  # fear-hoarding self-reinforcement              (D.5)
    c3=0.05,   # rival's arms into fear                        (D.5)
    c5=0.40,   # transparency drains fear                      (D.5)
    c6=0.10,   # natural dissipation of fear                   (D.5)
    rho=0.60,  # reserve replenishment per round               (D.3)
    kappa=1.0, # reserve cost of arms                          (D.3)
    cu=0.50,   # reserve cost of enforcement
    lam=0.15,  # reserve drawn by fear-driven hoarding         (D.3)
    m=0.30,    # arms restraint (Richardson)                   (11.2)
    R0=12.0,   # starting reserve
    umax=3.0,  # ceiling on enforcement in one round
    sig=0.05,  # observation noise
    Vwar=8.0,  # a play fails when violence passes this level
    H=60,      # rounds in a play
    pshock=0.12, shock_lo=0.4, shock_hi=1.2,   # injected grievance shocks
)

def shocks(rng, H, p):
    return np.where(rng.random(H) < p['pshock'], rng.uniform(p['shock_lo'], p['shock_hi'], H), 0.0)

def play(rule_a, rule_b, chi, tau_d, seed, p=P, log=False):
    """One play. rule = dict(K, tau_a, k_arm). Returns outcome dict (and logs)."""
    rng = np.random.default_rng(seed); H = p['H']
    V = np.zeros((2, H + 1)); F = np.zeros((2, H + 1)); A = np.zeros((2, H + 1)); R = np.zeros((2, H + 1))
    R[:, 0] = p['R0']; Y = np.zeros((2, H)); U = np.zeros((2, H)); UA = np.zeros((2, H))
    G = np.vstack([shocks(rng, H, p), shocks(rng, H, p)])
    rules = (rule_a, rule_b); failed = [False, False]; broke = [False, False]
    for t in range(H):
        for i in (0, 1):
            j = 1 - i; r = rules[i]
            td = max(t - tau_d, 0)
            # observation, equation (11.1): late, transparency-weighted, rumor fills the rest
            Y[i, t] = max(chi * V[i, td] + (1 - chi) * F[i, t] + p['sig'] * rng.standard_normal(), 0.0)
            ta = t - r['tau_a']
            want = r['K'] * Y[i, ta] if ta >= 0 else 0.0
            if r.get('noise'): want *= np.exp(r['noise'] * rng.standard_normal())   # an unsteady hand
            threat = chi * A[j, td] + (1 - chi) * F[i, t]
            want_arm = r['k_arm'] * threat
            u = min(want, p['umax']); ua = want_arm
            cost = p['cu'] * u + p['kappa'] * ua
            if cost > R[i, t]:                       # the reserve limits what can be done
                s = max(R[i, t], 0.0) / cost if cost > 0 else 0.0
                u *= s; ua *= s
            U[i, t] = u; UA[i, t] = ua
        for i in (0, 1):
            j = 1 - i
            V[i, t + 1] = max(V[i, t] + p['b1'] * G[i, t] + p['b2'] * V[i, t]
                              - (p['b3'] * U[i, t] * p['T'] + p['b4'] * p['T']) * V[i, t], 0.0)
            F[i, t + 1] = max(F[i, t] + p['c1'] * V[i, t] + p['c2h'] * F[i, t] + p['c3'] * A[j, t]
                              - (p['c5'] * chi + p['c6']) * F[i, t], 0.0)
            A[i, t + 1] = max(A[i, t] - p['m'] * A[i, t] + UA[i, t], 0.0)
            R[i, t + 1] = max(R[i, t] + p['rho'] - p['cu'] * U[i, t] - p['kappa'] * UA[i, t] - p['lam'] * F[i, t], 0.0)
            if V[i, t + 1] > p['Vwar']: failed[i] = True
            if R[i, t + 1] <= 0: broke[i] = True
        if failed[0] or failed[1]:
            V[:, t + 2:] = V[:, [t + 1]]; H_end = t + 1; break
    else:
        H_end = H
    out = dict(fail=failed[0] or failed[1], fail_a=failed[0], broke_a=broke[0], cumV=V[0, :H_end + 1].sum(), rounds=H_end)
    if log: out.update(V=V, F=F, A=A, R=R, Y=Y, U=U, UA=UA, G=G)
    return out

RULES = {
    'Prompt':       dict(K=1.5, tau_a=1, k_arm=0.10),
    'Slow':         dict(K=1.5, tau_a=4, k_arm=0.10),
    'Timid':        dict(K=0.4, tau_a=1, k_arm=0.10),
    'Overreacting': dict(K=6.0, tau_a=1, k_arm=0.10),
    'Arming':       dict(K=1.5, tau_a=1, k_arm=0.45),
}
CHIS = (0.9, 0.5, 0.2); N = 1000

def table(tau_d=1):
    print(f"\nShare of plays that fail (violence passes {P['Vwar']}), {N} plays per cell, sensing delay {tau_d} round(s)")
    print(f"{'team rule':14s}" + ''.join(f"  chi={c:<4}" for c in CHIS) + '   K   tau_a  k_arm')
    res = {}
    for name, r in RULES.items():
        row = []
        for c in CHIS:
            f = np.mean([play(r, r, c, tau_d, s)['fail'] for s in range(N)]); row.append(f)
        res[name] = row
        print(f"{name:14s}" + ''.join(f"  {x:8.2f}" for x in row) + f"   {r['K']:<4} {r['tau_a']:<5}  {r['k_arm']}")
    return res

def conditions():
    p = P
    print('\nLocal conditions with these coefficients')
    print(f"  (11.3) fear loop stable when c2h < c6 + c5*chi, that is chi > {(p['c2h'] - p['c6']) / p['c5']:.2f}")
    print(f"  (11.4) violence loop needs enforcement u_law > {(p['b2'] - p['b4'] * p['T']) / (p['b3'] * p['T']):.2f} to damp (open-loop coefficient b2 - b4*T = {p['b2'] - p['b4'] * p['T']:.2f})")
    print(f"  (11.5) delay bound tau_d + tau_a < 1/max(b2, c2h) = {1 / max(p['b2'], p['c2h']):.1f} rounds")
    print(f"  (11.2) arms pair stable when m*m' > k*k': m = {p['m']}, so k_arm < {p['m']}")

def delay_sweep(chi=0.9):
    print(f"\nFailure share against total delay tau_d + tau_a (K = 1.5, chi = {chi}); bound (11.5) is {1 / max(P['b2'], P['c2h']):.0f} rounds")
    for tot in range(2, 9):
        r = dict(K=1.5, tau_a=tot - 1, k_arm=0.10)
        f = np.mean([play(r, r, chi, 1, s)['fail'] for s in range(N)])
        print(f"  total delay {tot}: {f:.2f}")

def identification(chi=0.9, tau_d=1, n=300):
    """What a passive record gives, and what the experiment's log gives."""
    r = dict(RULES['Prompt'], noise=0.30); pas = []; b_open = []; b3 = []; Kh = []; th = []
    for s in range(n):
        o = play(r, r, chi, tau_d, 10_000 + s, log=True)
        V = o['V'][0]; U = o['U'][0]; Y = o['Y'][0]; G = o['G'][0]; H = o['rounds']
        if H < 30: continue
        v0, v1 = V[:H], V[1:H + 1]
        ok = v0 > 1e-6
        # passive record: only the violence series is seen
        pas.append(np.linalg.lstsq(np.c_[np.ones(ok.sum()), v0[ok]], v1[ok], rcond=None)[0][1] - 1)
        # experiment: shocks known, actions logged
        X = np.c_[G[:H], v0, U[:H] * P['T'] * v0]
        co = np.linalg.lstsq(X, v1 - v0, rcond=None)[0]
        b_open.append(co[1]); b3.append(-co[2])
        # decision rule from the log of what the team saw and did
        best = (-1, None, None)
        for lag in range(0, 7):
            y = Y[:H - lag]; u = U[lag:H]; keep = (u < P['umax'] - 1e-9) & (y > 0)
            if keep.sum() < 10: continue
            k = (y[keep] @ u[keep]) / (y[keep] @ y[keep]); r2 = 1 - ((u[keep] - k * y[keep]) ** 2).sum() / ((u[keep] - u[keep].mean()) ** 2).sum()
            if r2 > best[0]: best = (r2, lag, k)
        th.append(best[1]); Kh.append(best[2])
    f = lambda a: f"{np.mean(a):.3f} (sd {np.std(a):.3f})"
    print(f"\nIdentification, 'Prompt' teams whose actions scatter 30 percent around their rule, chi = {chi}, {len(pas)} plays")
    print(f"  true open-loop persistence b2 - b4*T = {P['b2'] - P['b4'] * P['T']:.3f}; true b3 = {P['b3']}; true K = {r['K']}; true tau_a = {r['tau_a']}")
    print(f"  passive record (violence series only), persistence estimate: {f(pas)}")
    print(f"  experiment log, open-loop persistence: {f(b_open)}")
    print(f"  experiment log, enforcement coefficient b3: {f(b3)}")
    print(f"  experiment log, team gain K: {f(Kh)}; delay tau_a recovered exactly in {np.mean(np.array(th) == r['tau_a']):.0%} of plays")

def learning(plays=60, teams=200):
    """A team that adjusts its gain between plays, keeping a change when its own play went better."""
    print(f"\nLearning teams: gain K starts at 0.4 and is revised after each play; {teams} teams, {plays} plays each")
    for chi in CHIS:
        first = []; last = []; Kend = []
        for g in range(teams):
            rng = np.random.default_rng(50_000 + g); K = 0.4; fails = []
            def loss(K, seed):
                o = play(dict(K=K, tau_a=1, k_arm=0.10), RULES['Prompt'], chi, 1, seed)
                return o['cumV'] + 50.0 * o['fail_a'] + 20.0 * o['broke_a'], o['fail_a']
            cur, fl = loss(K, 70_000 + 1000 * g)
            for n in range(plays):
                Kn = max(K * np.exp(0.35 * rng.standard_normal()), 0.05)
                new, fl = loss(Kn, 70_000 + 1000 * g + n + 1)
                fails.append(fl)
                if new < cur: K, cur = Kn, new
                else: cur = 0.7 * cur + 0.3 * loss(K, 70_000 + 1000 * g + n + 1)[0]
            first.append(np.mean(fails[:10])); last.append(np.mean(fails[-10:])); Kend.append(K)
        print(f"  chi = {chi}: failure share in first 10 plays {np.mean(first):.2f}, in last 10 plays {np.mean(last):.2f}; median final K {np.median(Kend):.2f}")

def sensitivity(draws=100):
    """Do the orderings survive when every coefficient is moved at random by up to 25 percent?"""
    rng = np.random.default_rng(7); keys = ['b1','b2','b3','b4','T','c1','c2h','c3','c5','c6','rho','kappa','cu','lam','m']
    a = b = c = 0; n = 200
    for d in range(draws):
        q = dict(P); 
        for k in keys: q[k] = P[k] * rng.uniform(0.75, 1.25)
        fr = lambda r, chi: np.mean([play(r, r, chi, 1, s, p=q)['fail'] for s in range(n)])
        hi, lo = fr(RULES['Prompt'], 0.9), fr(RULES['Prompt'], 0.2)
        arm = fr(RULES['Arming'], 0.9)
        o = play(RULES['Prompt'], RULES['Prompt'], 0.9, 1, 1, p=q, log=True); V = o['V'][0]; H = o['rounds']; ok = V[:H] > 1e-6
        pas = np.linalg.lstsq(np.c_[np.ones(ok.sum()), V[:H][ok]], V[1:H + 1][ok], rcond=None)[0][1] - 1 if ok.sum() > 5 else np.nan
        a += lo > hi; b += arm > hi; c += pas < q['b2'] - q['b4'] * q['T']
    print(f"\nSensitivity: every coefficient moved at random within 25 percent, {draws} draws")
    print(f"  low transparency fails more often than high: {a} of {draws}")
    print(f"  arming teams fail more often than prompt teams at chi = 0.9: {b} of {draws}")
    print(f"  passive record understates the open-loop persistence: {c} of {draws}")

if __name__ == '__main__':
    conditions(); table(1); table(3); delay_sweep(); identification(); learning(); sensitivity()
