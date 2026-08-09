"""
milling_force.py
================
Coefficients de force de fraisage hélicoïdal (Eqs. 3-4 de l'article).

Pour chaque dent j, l'angle de coupe vaut
    theta(t,j,z) = Omega*t + 2*pi*(j-1)/NT - z*tan(eta)/RT

Le modèle intègre les fonctions trigonométriques sur la portion
engagée [phi_st, phi_ex] et sur la profondeur axiale [za_low, za_high].
"""
import numpy as np


def cutting_coefficients(kn: float, mu_c: float, eta: float, gamma_n: float):
    """
    Coefficients k1, k2 of Eq.(3) of the article.

        k1 = kn / cos(eta)
        k2 = 1 + mu_c * tan(eta) * (cos(gamma_n) - kn * sin(gamma_n))

    kn      : proportionality constant  (0.26)
    mu_c    : friction coefficient      (0.20)
    eta     : helix angle       [rad]   (35 deg)
    gamma_n : normal rake angle [rad]   (15 deg)
    """
    k1 = kn / np.cos(eta)
    k2 = 1.0 + mu_c * np.tan(eta) * (np.cos(gamma_n) - kn * np.sin(gamma_n))
    return k1, k2


def milling_force_coeffs(t: float,
                         Omega: float, NT: int, RT: float, eta: float,
                         phi_st: float, phi_ex: float,
                         za_low: float, za_high: float,
                         k1: float, k2: float, kt: float):
    """
    Calcule (alpha3, alpha4) — coefficients de force de fraisage à l'instant t.
    """
    ss = 0.0
    sc = 0.0
    cc = 0.0
    te = np.tan(eta)

    for j in range(NT):
        theta0    = Omega*t + 2*np.pi*j/NT
        theta_max = theta0 - za_low *te/RT
        theta_min = theta0 - za_high*te/RT

        k_min = int(np.floor((theta_min - phi_ex)/(2*np.pi)))
        k_max = int(np.ceil ((theta_max - phi_st)/(2*np.pi)))

        for kk in range(k_min, k_max + 1):
            phi_lo = phi_st + 2*np.pi*kk
            phi_hi = phi_ex + 2*np.pi*kk

            th_lo = max(theta_min, phi_lo)
            th_hi = min(theta_max, phi_hi)

            if th_hi <= th_lo:
                continue

            z2 = (theta0 - th_lo)*RT/te
            z1 = (theta0 - th_hi)*RT/te
            z1 = max(z1, za_low)
            z2 = min(z2, za_high)

            if z2 <= z1:
                continue

            theta1 = theta0 - z1*te/RT
            theta2 = theta0 - z2*te/RT

            ss += (z2 - z1)/2 \
                + RT/(2*te) * np.sin(theta2 - theta1) * np.cos(theta2 + theta1)
            sc += RT/(2*te) * np.sin(theta1 - theta2) * np.sin(theta1 + theta2)
            cc += (z2 - z1)/2 \
                - RT/(2*te) * np.sin(theta2 - theta1) * np.cos(theta2 + theta1)

    alpha3 =  k2*kt*ss - k1*kt*sc
    alpha4 =  k2*kt*sc - k1*kt*cc
    return alpha3, alpha4


def precompute_alpha_periodic(dt: float, n_per: int, n_total: int,
                              Omega: float, NT: int, RT: float, eta: float,
                              phi_st: float, phi_ex: float,
                              za_low: float, za_high: float,
                              k1: float, k2: float, kt: float):
    """
    Calcule alpha3(t), alpha4(t) sur une période tau (n_per pas) puis duplique.
    Exploite la périodicité tau = 60/(NT*RPM) pour gros gain en vitesse.
    """
    alpha3_per = np.zeros(n_per)
    alpha4_per = np.zeros(n_per)
    for k in range(n_per):
        alpha3_per[k], alpha4_per[k] = milling_force_coeffs(
            k*dt, Omega, NT, RT, eta,
            phi_st, phi_ex, za_low, za_high, k1, k2, kt
        )

    nrep = n_total // n_per + 2
    alpha3_t = np.tile(alpha3_per, nrep)[:n_total]
    alpha4_t = np.tile(alpha4_per, nrep)[:n_total]
    return alpha3_t, alpha4_t
