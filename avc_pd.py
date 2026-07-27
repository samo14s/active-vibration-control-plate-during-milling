"""
PD control law replacing the two-mode PPF.

    u = -( kp + kd * s / (1 + s/wc) ) * y

The derivative is filtered (real pole at wc) so the law stays proper and does
not amplify sensor noise or the truncated high-frequency modes.

Structural property worth stating: the patch sensor is collocated with the
patch actuator, so cS = bmod and the feedback adds

    dK = kp * bmod bmod^T   (stiffness)      dC = kd * bmod bmod^T   (damping)

Both are rank-1 and positive semi-definite, so a negative-feedback PD is
passive on this pair -- it cannot destabilise the plant at any gain (the only
stability limit comes from the derivative filter pole).  Contrast with PPF,
which is POSITIVE position feedback: it subtracts stiffness and therefore
degrades the static regime.  Here the P term adds stiffness instead.

Tuned by the same objective and the same constraints as the PPF baseline:
maximise the worst-tool-position stability-limit lift subject to
peak effort <= VLIM V/N and no mode below ZMIN.
"""
import numpy as np
from avc_plant import Plant, Ctrl, score, VLIM, ZMIN


def pd_ctrl(kp, kd, wc, name='PD'):
    """State-space realisation of -(kp + kd s/(1+s/wc))."""
    Ak = np.array([[-wc]])
    Bk = np.array([1.0])
    Ck = np.array([kd * wc ** 2])
    Dk = -(kp + kd * wc)
    return Ctrl(Ak, Bk, Ck, Dk, name)


FC_D = 10e3          # Hz -- derivative filter cutoff, implementable and within 5 %
                     # of the unconstrained optimum (see fc sweep in the report)


def design(plant, wc=2 * np.pi * FC_D, verbose=True, seed=1):
    def obj(theta):
        kp, kd = np.exp(theta)
        K = pd_ctrl(kp, kd, wc)
        r = score(plant, K)
        if not r['stable']:
            return -1.0
        if r['zeta_min'] < ZMIN:
            return -1.0
        if r['volt'] > VLIM:
            return -1.0
        return r['a_lim']

    best = (-1.0, None)
    for lkp in np.log(np.logspace(5, 11, 13)):
        for lkd in np.log(np.logspace(3, 7, 13)):
            th = np.array([lkp, lkd])
            s = obj(th)
            if s > best[0]:
                best = (s, th)
    if best[1] is None:
        raise RuntimeError('no feasible PD found')

    th = best[1].copy()
    rng = np.random.default_rng(seed)
    for sc in [0.5, 0.2, 0.08, 0.03, 0.01]:
        for _ in range(120):
            cand = th + rng.standard_normal(2) * sc
            s = obj(cand)
            if s > best[0]:
                best = (s, cand); th = cand
    kp, kd = np.exp(best[1])
    if verbose:
        print('PD  kp = %.3e V/m   kd = %.3e V.s/m   fc = %.0f Hz   -> a_lim x%.2f'
              % (kp, kd, wc / 2 / np.pi, best[0]))
    return pd_ctrl(kp, kd, wc, 'PD'), dict(kp=kp, kd=kd, wc=wc)


if __name__ == '__main__':
    P = Plant()
    K, par = design(P)
    r = score(P, K)
    print('a_lim x%.2f  volt %.1f V/N  peak %.1f um/N  z1 %.2f%%  z2 %.2f%%  static ratio %.3f'
          % (r['a_lim'], r['volt'], r['peak_cl'] * 1e6, r['zeta1'], r['zeta2'], r['static_ratio']))
