"""Classical modal model of the thin-walled cantilever plate.

This is the FEM-free replacement for avc.fem_plate + avc.modal. The plate
is a rectangular CFFF cantilever (clamped along the bottom edge z = 0,
free on the other three edges, milled along the top free edge z = h), and
its normal (out-of-plane, y) vibration is represented in the classical
machining-dynamics way: a small set of separable normal modes with
frequencies identified from the reference rig's experimental modal
analysis (Du et al. 2024, Table 4) and mode shapes given by the classical
Warburton beam-product construction

    phi_{mn}(x, z) = X_m(x) * Z_n(z),   0 <= x <= a (feed),  0 <= z <= b,

with X_m the free-free beam eigenfunctions along the feed direction
(both x-edges free) and Z_n the clamped-free (cantilever) beam
eigenfunctions along the span (clamped at z = 0). The per-mode plate
natural frequency is the classical Rayleigh quotient of the thin-plate
strain energy evaluated on that product (Warburton 1954); the working
model anchors the first five frequencies to the measured values and lets
the analytical Rayleigh ratio propagate material removal.

The object returned by :func:`reduce_model_classical` duck-types
avc.modal.ModalModel (same ``omega``/``zeta``/``bu``/``cs``/``bf``/
``abcd``/``frf_voltage``/``freqs_hz`` interface), so the entire synthesis,
stability, certification and simulation stack runs on it unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .params import Params, milling_point

# ---------------------------------------------------------------------------
# 1. Euler-Bernoulli beam eigenfunctions (analytic, on [0, L])
# ---------------------------------------------------------------------------
# Frequency roots kL of the two end conditions we need. Clamped-free:
# cos(k)cosh(k) = -1; free-free (elastic): cos(k)cosh(k) = +1. The first
# handful are hard-coded (Newton-polished below); the asymptotic
# (2i-1)pi/2 seeds the rest.
_CF_SEED = (1.875104, 4.694091, 7.854757, 10.995541, 14.137168)
_FF_SEED = (4.730041, 7.853205, 10.995608, 14.137165, 17.278759)


def _polish(k0: float, clamped_free: bool) -> float:
    """Newton-refine a root of cos(k)cosh(k) -/+ 1."""
    k = float(k0)
    s = -1.0 if clamped_free else 1.0
    for _ in range(60):
        c, ch = np.cos(k), np.cosh(k)
        f = c * ch - s
        df = -np.sin(k) * ch + c * np.sinh(k)
        step = f / df
        k -= step
        if abs(step) < 1e-14:
            break
    return k


@dataclass
class _BeamMode:
    """One clamped-free or free-free beam eigenfunction on [0, L]."""
    L: float
    k: float                 # = beta * L (0 for a free-free rigid mode)
    sigma: float
    kind: str                # 'cf', 'ff', 'ff_t' (translation), 'ff_r' (rot)

    def _u(self, s, d):
        """d-th derivative (d=0,1,2) of the shape at physical coord s."""
        b = self.k / self.L
        xi = b * np.asarray(s, float)
        if self.kind == "ff_t":                       # rigid translation
            return np.ones_like(xi) if d == 0 else np.zeros_like(xi)
        if self.kind == "ff_r":                       # rigid rotation
            if d == 0:
                return np.sqrt(3.0) * (2.0 * s / self.L - 1.0)
            if d == 1:
                return np.sqrt(3.0) * (2.0 / self.L) * np.ones_like(xi)
            return np.zeros_like(xi)
        ch, sh, co, si = np.cosh(xi), np.sinh(xi), np.cos(xi), np.sin(xi)
        if self.kind == "cf":                         # clamped-free
            if d == 0:
                f = (ch - co) - self.sigma * (sh - si)
            elif d == 1:
                f = b * ((sh + si) - self.sigma * (ch - co))
            else:
                f = b * b * ((ch + co) - self.sigma * (sh + si))
        else:                                         # free-free elastic
            if d == 0:
                f = (ch + co) - self.sigma * (sh + si)
            elif d == 1:
                f = b * ((sh - si) - self.sigma * (ch + co))
            else:
                f = b * b * ((ch - co) - self.sigma * (sh - si))
        return f

    def val(self, s):
        return self._u(s, 0)

    def d2(self, s):
        return self._u(s, 2)


def _beam_modes(L: float, kind: str, n: int) -> list[_BeamMode]:
    """Lowest n beam eigenfunctions of the given end condition on [0, L].

    kind='cf' -> clamped-free; kind='ff' -> free-free (the two rigid-body
    modes first, then elastic).
    """
    out: list[_BeamMode] = []
    if kind == "ff":
        out.append(_BeamMode(L, 0.0, 0.0, "ff_t"))
        out.append(_BeamMode(L, 0.0, 0.0, "ff_r"))
        seeds = _FF_SEED
    else:
        seeds = _CF_SEED
    i = 0
    while len(out) < n:
        if i < len(seeds):
            k = _polish(seeds[i], kind == "cf")
        else:
            k = _polish((2 * (i + 1) - 1) * np.pi / 2.0, kind == "cf")
        if kind == "cf":
            sigma = (np.cosh(k) + np.cos(k)) / (np.sinh(k) + np.sin(k))
            out.append(_BeamMode(L, k, sigma, "cf"))
        else:
            sigma = (np.cosh(k) - np.cos(k)) / (np.sinh(k) - np.sin(k))
            out.append(_BeamMode(L, k, sigma, "ff"))
        i += 1
    return out[:n]


# ---------------------------------------------------------------------------
# 2. quadrature helpers (dense Gauss-Legendre on the beam span)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _gl(L: float, n: int = 400):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * L * (x + 1.0), 0.5 * L * w      # nodes, weights on [0, L]


def _mode_integrals(m: _BeamMode):
    """(I0, I1region-free) integrals over [0, L]: I0=int f^2, I2=int f''^2,
    Ix=int f'^2, Ic=int f f''  (all needed by the Rayleigh quotient)."""
    s, w = _gl(m.L)
    f = m.val(s)
    f1 = m._u(s, 1)
    f2 = m.d2(s)
    return (float(w @ (f * f)), float(w @ (f2 * f2)),
            float(w @ (f1 * f1)), float(w @ (f * f2)))


def _patch_integrals(m: _BeamMode, lo: float, hi: float):
    """(int f dx, int f'' dx) over the sub-interval [lo, hi]."""
    s, w = _gl(m.L)
    inside = (s >= lo) & (s <= hi)
    return float(w[inside] @ m.val(s[inside])), \
        float(w[inside] @ m.d2(s[inside]))


# ---------------------------------------------------------------------------
# 3. classical plate modal model (drop-in for avc.modal.ModalModel)
# ---------------------------------------------------------------------------
@dataclass
class ClassicalModalModel:
    params: Params
    height_removed: float
    n_modes: int
    omega: np.ndarray
    zeta: np.ndarray
    bu: np.ndarray                 # modal piezo input Phi^T Theta [N/V]
    cs: np.ndarray                 # sensor output row  w(sensor)
    _Xk: list = field(default_factory=list)   # per-mode X_m beam mode
    _Zk: list = field(default_factory=list)   # per-mode Z_n beam mode
    _norm: np.ndarray = None       # per-mode mass-normalization 1/sqrt(rho h I)
    height: float = 0.0
    model = None                   # no FEM; kept for interface parity

    @property
    def freqs_hz(self) -> np.ndarray:
        return self.omega / (2.0 * np.pi)

    # ---- mass-normalized mode shape value at a point ----------------------
    def _phi(self, i: int, x: float, z: float) -> float:
        return float(self._Xk[i].val(x) * self._Zk[i].val(z) * self._norm[i])

    def bf(self, x_T: float) -> np.ndarray:
        """Modal participation of the milling point (top free edge)."""
        z = min(milling_point(self.params, self.height_removed), self.height)
        return np.array([self._phi(i, x_T, z) for i in range(self.n_modes)])

    def abcd(self, x_T: float, alpha4: float = 0.0):
        """Identical construction to avc.modal.ModalModel.abcd (see there
        for the load-bearing sign convention on ``alpha4``)."""
        n = self.n_modes
        bf = self.bf(x_T)
        W2 = np.diag(self.omega ** 2)
        Z = np.diag(2.0 * self.zeta * self.omega)
        A = np.block([[np.zeros((n, n)), np.eye(n)],
                      [-(W2 - alpha4 * np.outer(bf, bf)), -Z]])
        Bu = np.concatenate([np.zeros(n), self.bu])[:, None]
        Bf = np.concatenate([np.zeros(n), bf])[:, None]
        Cs = np.concatenate([self.cs, np.zeros(n)])[None, :]
        Cw = np.concatenate([bf, np.zeros(n)])[None, :]
        return A, Bu, Bf, Cs, Cw

    def frf_voltage(self, freqs_hz: np.ndarray, x_out: float,
                    z_out: float) -> np.ndarray:
        """Voltage -> displacement FRF at a point [m/V]."""
        c = np.array([self._phi(i, x_out, z_out) for i in range(self.n_modes)])
        w = 2.0 * np.pi * np.asarray(freqs_hz, float)
        H = np.zeros(len(w), dtype=complex)
        for i in range(self.n_modes):
            H += c[i] * self.bu[i] / (self.omega[i] ** 2 - w ** 2
                                      + 2j * self.zeta[i] * self.omega[i] * w)
        return H


def _enumerate_modes(a: float, b: float, params: Params, n_need: int):
    """Rayleigh frequencies + shapes of the lowest ~n_need product modes.

    a = plate length (feed x, free-free), b = span (clamped-free z).
    Returns sorted lists (omega_rayleigh, X_mode, Z_mode).
    """
    pl = params.plate
    D = pl.E * pl.bp ** 3 / (12.0 * (1.0 - pl.nu ** 2))
    mu = pl.rho * pl.bp                       # mass per area
    nu = pl.nu
    # generous product grid; keep the lowest n_need after sorting
    nx = max(4, n_need)
    nz = max(6, n_need)
    Xs = _beam_modes(a, "ff", nx)
    Zs = _beam_modes(b, "cf", nz)
    XI = [_mode_integrals(X) for X in Xs]     # (I0, I2, Ix, Ic)
    ZI = [_mode_integrals(Z) for Z in Zs]
    cand = []
    for im, X in enumerate(Xs):
        X0, X2, Xx, Xc = XI[im]
        for iz, Z in enumerate(Zs):
            Z0, Z2, Zx, Zc = ZI[iz]
            # thin-plate strain-energy Rayleigh quotient of X(x)Z(z)
            U = (X2 * Z0 + X0 * Z2 + 2.0 * nu * Xc * Zc
                 + 2.0 * (1.0 - nu) * Xx * Zx)
            T = X0 * Z0                       # int phi^2 (times mu below)
            if T <= 0 or U <= 0:
                continue
            w2 = D * U / (mu * T)
            cand.append((np.sqrt(w2), im, iz, X0 * Z0))
    cand.sort(key=lambda c: c[0])
    return cand[:n_need], Xs, Zs, XI, ZI


@lru_cache(maxsize=64)
def reduce_model_classical(params: Params, n_modes: int,
                           height_removed: float = 0.0) -> ClassicalModalModel:
    """Classical Warburton modal model of the receded plate.

    Frequencies of the first five modes are anchored to the measured EMA
    values; material removal and higher modes use the analytical Rayleigh
    ratio. All spatial couplings (piezo, sensor, tool point) come from the
    beam-product mode shapes.
    """
    pl = params.plate
    a = pl.lp
    b0 = pl.hp
    b = pl.hp - height_removed
    pz = params.piezo

    cand, Xs, Zs, XI, ZI = _enumerate_modes(a, b, params, n_modes)
    # nominal-geometry Rayleigh set (same product ordering) supplies the
    # analytical ratio that propagates material removal onto the measured
    # frequency anchors
    cand_nom, Xs0, Zs0, XI0, ZI0 = _enumerate_modes(a, b0, params, n_modes)

    f_meas = list(pl.f_measured)
    m1 = pz.E / (1.0 - pz.nu) * pz.d31 * 0.5 * (pl.bp + pz.h_pa)
    mu = pl.rho * pl.bp

    omega = np.zeros(n_modes)
    zeta = np.zeros(n_modes)
    bu = np.zeros(n_modes)
    cs = np.zeros(n_modes)
    Xk, Zk, norm = [], [], np.zeros(n_modes)
    z_top = min(milling_point(params, height_removed), b)

    for r in range(n_modes):
        w_ray, im, iz, I00 = cand[r]
        w_nom = cand_nom[r][0]
        X, Z = Xs[im], Zs[iz]
        # frequency: measured anchor (first 5) scaled by the removal ratio;
        # analytical Rayleigh for the higher spillover-evaluation modes
        if r < len(f_meas):
            omega[r] = 2.0 * np.pi * f_meas[r] * (w_ray / w_nom)
        else:
            omega[r] = w_ray
        # mass normalization: int rho h phi^2 dA = 1
        norm[r] = 1.0 / np.sqrt(mu * I00)
        # piezo modal input: m1 * int_patch (phi_xx + phi_zz) dA
        Xp_f, Xp_2 = _patch_integrals(X, pz.x0, pz.x0 + pz.length)
        Zp_f, Zp_2 = _patch_integrals(Z, pz.z0, pz.z0 + pz.width)
        Z0, _, _, _ = ZI[iz]
        X0, _, _, _ = XI[im]
        curv = Xp_2 * Zp_f + Xp_f * Zp_2      # int (X''Z + X Z'') over patch
        bu[r] = m1 * curv * norm[r]
        # sensor row
        cs[r] = X.val(params.sensor.xs) * Z.val(
            min(params.sensor.zs, b)) * norm[r]
        Xk.append(X)
        Zk.append(Z)

    zeta = _classical_damping(params, n_modes)
    return ClassicalModalModel(
        params=params, height_removed=height_removed, n_modes=n_modes,
        omega=omega, zeta=zeta, bu=bu, cs=cs, _Xk=Xk, _Zk=Zk, _norm=norm,
        height=b)


def _classical_damping(params: Params, n: int) -> np.ndarray:
    z = list(params.plate.zeta)
    if n <= len(z):
        return np.array(z[:n])
    return np.array(z + [params.plate.zeta_high] * (n - len(z)))
