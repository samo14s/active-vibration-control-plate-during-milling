"""
Exemple d'interface — comment brancher un correcteur sur la base.

Le correcteur ci-dessous est un simple retour de vitesse. Il NE STABILISE PAS
cette configuration, et c'est instructif : le patch et le capteur sont aux coins
OPPOSES de la plaque, donc la paire n'est pas colocalisee et le retour de vitesse
direct ne beneficie d'aucune garantie de passivite. Sur cette plaque il degrade
la limite de stabilite au lieu de l'ameliorer.

Ce fichier sert donc a deux choses : montrer les quatre appels de l'API, et
donner un point de comparaison bas que tout correcteur serieux doit battre.

    python example_controller.py
"""
import numpy as np
from simulation_base import SimBase, check_model, V_MAX


class VelocityFeedback:
    """u = -k * dy/dt, sature. Un seul reglage : k."""

    def __init__(self, dt, k):
        self.dt = dt
        self.k = k
        self.y_prev = 0.0

    def step(self, x_hat_prev, u_prev, y_meas):
        v = (y_meas - self.y_prev)/self.dt
        self.y_prev = y_meas
        return x_hat_prev, float(np.clip(-self.k*v, -V_MAX, V_MAX))

    def reset(self):
        self.y_prev = 0.0


if __name__ == "__main__":
    sim = check_model(verbose=False)
    print("base verifiee\n")

    print("APPEL 1 — un essai unique, coupe stable a 0.05 mm")
    r = sim.run(None, rpm=4900, ap=0.05e-3, T=0.20)
    print(f"   sans controle : stable={r['stable']}  RMS={r['rms_um']:.4f} um  "
          f"croissance={r['growth']:.3f}")
    r = sim.run(VelocityFeedback(sim.run(None, ap=0.05e-3)['dt'], 5.0),
                rpm=4900, ap=0.05e-3, T=0.20)
    print(f"   avec k = 5    : stable={r['stable']}  RMS={r['rms_um']:.4f} um  "
          f"RMS u={r['rms_u']:.3f} V\n")

    print("APPEL 2 — cout multi-vitesses (a minimiser)")
    for k in (0.0, 5.0, 20.0):
        J = sim.multi_speed_cost(lambda dt, tau, k=k: VelocityFeedback(dt, k))
        print(f"   k = {k:5.1f} : J = {J:8.4f}")
    print("   (J = 12 signifie instable aux cinq vitesses)\n")

    print("APPEL 3 — limite de stabilite a 4900 tr/min")
    lo = sim.stability_limit(None, rpm=4900)
    print(f"   sans controle : {lo*1e3:.4f} mm")
    for k in (5.0, 20.0):
        b = sim.stability_limit(lambda dt, tau, k=k: VelocityFeedback(dt, k),
                                rpm=4900)
        print(f"   k = {k:5.1f}     : {b*1e3:.4f} mm   (facteur {b/lo:.2f})")
    print("   le retour de vitesse direct DEGRADE ici : paire non colocalisee\n")

    print("APPEL 4 — reponse frequentielle mesuree")
    f, G0 = sim.receptance(None)
    i = int(np.argmin(abs(f - 536.4)))
    print(f"   sans controle, pic a 536 Hz : {G0[i]:8.2f} um/N")
    f, G1 = sim.receptance(lambda dt, tau: VelocityFeedback(dt, 5.0))
    if G1 is None:
        print("   avec controle : divergence")
    else:
        print(f"   avec k = 5                  : {G1[i]:8.2f} um/N   "
              f"({20*np.log10(G1[i]/G0[i]):+.1f} dB)")

    print("\nreperes a battre : limite 0.0605 mm, pic 488 um/N, J = 11.15 sans controle")
