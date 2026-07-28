"""
Driver for the new law: removal-adaptive, frontier-certified convex chatter
control.  Produces everything the comparison and the manuscript need:

  1. drift sweep -- the guaranteed floor vs the frequency drift the law
     must ride out between two convex re-solves ("the price of robustness");
  2. per-stage designs (fresh / mid / machined) at the working drift,
     realised, reduced, scored, ZOA floors verified from the realised K;
  3. authority frontier per stage (the controller-independent ceiling);
  4. speed-scheduled designs at the paper's test speeds;
  5. FDM verification: SLDs of uncontrolled vs the new law (stage-matched),
     spot checks of the scheduled variant;
  6. time-domain showcase at the paper's condition S (4900 rpm, 0.3 mm).

Run:  python3 -u avc_newlaw.py            (background; writes newlaw_log.txt
                                           by the caller's redirection and
                                           _newlaw_*.npz artefacts)
"""
import numpy as np
import control as ct
import milling_model as mm
import avc_frontier as af
from avc_frontier import UM
from avc_plant import Plant, Ctrl, score, close
import avc_fdm as fdm

KAP = abs(mm.kappa4())
VMARGIN = 140.0                  # design-time budget; verified against 150
RPM_GRID = np.linspace(3000.0, 7000.0, 41)
RPM_TEST = [3600.0, 4900.0, 5400.0]

STAGES = [('fresh', np.ones(6)),
          ('mid', 1.0 + 0.5 * (mm.REMOVAL_PAPER - 1.0)),
          ('machined', mm.REMOVAL_PAPER)]


def dense_grid():
    """Verification grid: log + clusters around every mode of every stage."""
    base = np.geomspace(60.0, 5200.0, 340)
    Ps = [Plant(zeta=mm.ZETA_PAPER, freq_scale=s) for _, s in STAGES]
    cl = []
    for P in Ps:
        for f in P.f:
            if f < 5200:
                cl.append(np.linspace(0.90 * f, 1.10 * f, 25))
    fr = np.unique(np.concatenate([base] + cl))
    return 2 * np.pi * fr


def synth_grid():
    """Synthesis grid: the PD-damped FRFs are smooth (percent-wide
    resonances), so ~500 points carry the same information."""
    base = np.geomspace(60.0, 5200.0, 260)
    Ps = [Plant(zeta=mm.ZETA_PAPER, freq_scale=s) for _, s in STAGES]
    cl = []
    for P in Ps:
        for f in P.f:
            if f < 5200:
                cl.append(np.linspace(0.92 * f, 1.08 * f, 13))
    fr = np.unique(np.concatenate([base] + cl))
    return 2 * np.pi * fr


def zoa_floor_of_K(P, K, om):
    """Guaranteed broadband floor of a realised controller, worst position."""
    worst = 0.0
    for x in af.POSITIONS:
        D = P.edge_D(x)
        A, Bw, Cz, Cv = close(P, K, D)
        lam, V = np.linalg.eig(A)
        W = np.linalg.solve(V, Bw[:, 0])
        CzV = (Cz @ V)[0]
        G = ((CzV * W)[None, :] / (1j * om[:, None] - lam[None, :])).sum(1)
        worst = max(worst, float((np.abs(G) - G.real).max()))
    return 1.0 / (KAP * worst)


def reduce_ctrl(K, P, om, target=24):
    """Balanced truncation of the realised law, accepted only if the floor
    it guarantees moves by < 2 %."""
    sys = ct.ss(K.Ak, K.Bk, K.Ck, K.Dk)
    f_full = zoa_floor_of_K(P, K, om)
    for r in range(target, 61, 6):
        try:
            red = ct.balred(sys, r)
        except Exception:
            continue
        Ar, Br, Cr, Dr = ct.ssdata(red)
        Kr = Ctrl(Ar, Br.ravel(), Cr.ravel(), float(Dr[0, 0]), K.name)
        f_red = zoa_floor_of_K(P, Kr, om)
        if abs(f_red - f_full) < 0.02 * f_full:
            return Kr, r, f_red
    return K, K.order, f_full


def main():
    from avc_pd import pd_ctrl
    par = np.load('_pd.npz')
    K_in = pd_ctrl(float(par['kp']), float(par['kd']), float(par['wc']),
                   'PD inner')
    omv = dense_grid()
    oms = synth_grid()
    print('grids: synth %d pts, verify %d pts' % (len(oms), len(omv)),
          flush=True)
    P0 = Plant(zeta=mm.ZETA_PAPER)
    f_unc = zoa_floor_of_K(P0, Ctrl(np.zeros((0, 0)), [], [], 0.0), omv)
    f_pd = zoa_floor_of_K(P0, K_in, omv)
    print('ZOA floors: uncontrolled %.4f mm   PD inner alone %.4f mm'
          % (f_unc * 1e3, f_pd * 1e3), flush=True)

    # ---------------------------------------------------- 1. drift sweep
    print('=== drift sweep (fresh stage, damped layer) ===', flush=True)
    drift_out = []
    for drift in [0.005, 0.01, 0.02, 0.04]:
        dat = af.stage_data_damped(K_in, om=oms, drift=drift)
        th, c, floor, bank = af.synth_floor(dat, vbar=VMARGIN, verbose=False)
        drift_out.append((drift, floor))
        print('  drift %.1f%% : guaranteed floor %.4f mm'
              % (100 * drift, floor * 1e3), flush=True)
    np.savez('_newlaw_drift.npz', drift=np.array(drift_out),
             f_unc=f_unc, f_pd=f_pd)

    # ---------------------------------------- 2-3. per-stage designs
    DRIFT = 0.01
    stages_out = {}
    for tag, scale in STAGES:
        print('=== stage %s ===' % tag, flush=True)
        P = Plant(zeta=mm.ZETA_PAPER, freq_scale=scale)
        datu = af.stage_data(stage_scale=scale, om=oms, drift=DRIFT)
        ap_star, _ = af.frontier(datu)
        dat = af.stage_data_damped(K_in, stage_scale=scale, om=oms,
                                   drift=DRIFT)
        th, c, floor, bank = af.synth_floor(dat, vbar=VMARGIN, verbose=True)
        K = af.realise_damped(th, bank, P, K_in)
        Kr, order, f_red = reduce_ctrl(K, P, omv)
        r = score(P, Kr)
        print('  SOCP floor %.4f mm | frontier %.3f mm (%.0f %%) | order %d->%d'
              % (floor * 1e3, ap_star * 1e3, 100 * floor / ap_star,
                 K.order, order), flush=True)
        print('  realised: floor %.4f mm  volt %.1f V/N  stable %s  zmin %.2f%%'
              % (f_red * 1e3, r['volt'], r['stable'], r['zeta_min'] * 100),
              flush=True)
        stages_out[tag] = dict(theta=th, c=c, floor=floor, frontier=ap_star,
                               floor_real=f_red, order=order,
                               Ak=Kr.Ak, Bk=Kr.Bk, Ck=Kr.Ck, Dk=Kr.Dk,
                               volt=r['volt'], zmin=r['zeta_min'])
        np.savez('_newlaw_stage_%s.npz' % tag, **stages_out[tag])

    # ------------------------------------------ 4. scheduled designs
    print('=== speed-scheduled designs (fresh stage) ===', flush=True)
    dat = af.stage_data_damped(K_in, om=oms, drift=DRIFT)
    sched = {}
    for rpm in RPM_TEST:
        th_s, ap_s = af.synth_scheduled(dat, rpm, vbar=VMARGIN)
        sched[int(rpm)] = (th_s, ap_s)
        print('  %4.0f rpm : ZOA scheduled ap %.3f mm' % (rpm, ap_s * 1e3),
              flush=True)
    np.savez('_newlaw_sched.npz',
             **{'rpm_%d' % r: np.concatenate([sched[r][0], [sched[r][1]]])
                for r in sched})

    # ------------------------------------------ 5. FDM verification
    print('=== FDM verification (worst position x = 100 mm) ===', flush=True)
    d = stages_out['fresh']
    K0 = Ctrl(d['Ak'], d['Bk'].ravel(), d['Ck'].ravel(), float(d['Dk']),
              'passive + convex')
    xw = 100.0
    D = P0.edge_D(xw)
    sld_unc = fdm.fdm_sld(P0, D, RPM_GRID, None, ap_hi=4e-3, m=40)
    print('  uncontrolled SLD: min %.3f mm' % (sld_unc.min() * 1e3), flush=True)
    sld_pd = fdm.fdm_sld(P0, D, RPM_GRID, K_in, ap_hi=8e-3, m=40)
    print('  PD inner SLD:     min %.3f mm' % (sld_pd.min() * 1e3), flush=True)
    sld_new = fdm.fdm_sld(P0, D, RPM_GRID, K0, ap_hi=8e-3, m=40)
    print('  new law SLD:      min %.3f mm  (lift x%.1f over uncontrolled)'
          % (sld_new.min() * 1e3, sld_new.min() / max(sld_unc.min(), 1e-12)),
          flush=True)
    np.savez('_newlaw_sld.npz', rpm=RPM_GRID, unc=sld_unc, pd=sld_pd,
             new=sld_new)

    Pm = Plant(zeta=mm.ZETA_PAPER, freq_scale=mm.REMOVAL_PAPER)
    dm = stages_out['machined']
    Km = Ctrl(dm['Ak'], dm['Bk'].ravel(), dm['Ck'].ravel(), float(dm['Dk']))
    Dm = Pm.edge_D(xw)
    sld_m = fdm.fdm_sld(Pm, Dm, RPM_GRID, Km, ap_hi=8e-3, m=40)
    print('  machined, stage-matched law: min %.3f mm' % (sld_m.min() * 1e3),
          flush=True)
    sld_x = fdm.fdm_sld(Pm, Dm, RPM_GRID, K0, ap_hi=8e-3, m=40)
    print('  machined, FROZEN fresh law:  min %.3f mm' % (sld_x.min() * 1e3),
          flush=True)
    np.savez('_newlaw_sld2.npz', rpm=RPM_GRID, machined=sld_m, cross=sld_x)

    # ------------------------------------------ 6. time-domain showcase
    print('=== time sim at S (4900 rpm, ap 0.3 mm) ===', flush=True)
    out = {}
    for name, KK in [('uncontrolled', None), ('pd', K_in), ('new', K0)]:
        t, y, u = fdm.milling_sim(P0, D, 4900.0, 0.3e-3, KK, T_end=0.35)
        out[name + '_t'], out[name + '_y'], out[name + '_u'] = t, y, u
        print('  %-12s |y|max %.1f um   |u|max %.1f V'
              % (name, np.abs(y).max() * 1e6, np.abs(u).max()), flush=True)
    np.savez('_newlaw_sim.npz', **out)
    print('newlaw driver done', flush=True)


if __name__ == '__main__':
    main()
