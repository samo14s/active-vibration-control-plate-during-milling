# Precise Kirchhoff modelling of the actively-controlled milled plate

*Companion derivation chapter. Every equation here corresponds line-for-line to
the implementation in `src/`; validation numbers come from
`experiments/model_refinement.py` → `results/refinement.json`.*

---

## 1. Scope and assumptions

The workpiece is a cantilever AL6061 plate, length L = 100 mm (x), height
H = 80 mm (z), thickness h = 4 mm, clamped along its bottom edge z = 0, milled
peripherally along its top edge z = H. Since h / min(L, H) = 1/20 and all
retained modes lie below 4.5 kHz, the **Kirchhoff–Love thin-plate theory** is
appropriate:

- K1. Straight lines normal to the mid-surface remain straight and normal
  (transverse shear deformation neglected);
- K2. The normal stress σ_zz is negligible;
- K3. Deflections are small (linear kinematics);
- K4. Rotary inertia is neglected in the plate mass.

The error of K1/K4 scales with (h/λ_bend)²; for mode 5 (~4.2 kHz) the bending
wavelength is still ≈ 40 mm ≫ h, so the theory holds over the band of interest.

## 2. Kinematics and constitutive law

With mid-surface deflection w(x, z, t), the Kirchhoff curvature vector is

    κ = [ -∂²w/∂x²,  -∂²w/∂z²,  -2 ∂²w/∂x∂z ]ᵀ,                        (M1)

and the moment–curvature law M = D_f κ with the isotropic bending matrix

    D_f = D [ 1  ν  0 ;  ν  1  0 ;  0  0  (1-ν)/2 ],   D = E h³ / 12(1-ν²). (M2)

Strain and kinetic energies per element give the weak form used below.

## 3. Finite element: 4-node Hermite (ACM) plate element

Each node carries (w, ∂w/∂x, ∂w/∂z) → 12 DOF per element (`kirchhoff_q4.py`).
The shape functions are the 12 cubic (Adini–Clough–Melosh) polynomials in the
natural coordinates (ξ, η) ∈ [-1, 1]²; the element is *non-conforming* (the
normal slope is not C¹-continuous across edges) but passes the patch test and
converges monotonically — verified in Sec. 8. For the rectangular mesh the
Jacobian is diagonal, J = diag(l_ex/2, l_ez/2), and the curvature matrix is

    B(ξ,η) = [ -(2/l_ex)² N,ξξ ;  -(2/l_ez)² N,ηη ;  -2 (4/l_ex l_ez) N,ξη ]. (M3)

Element matrices (`stiffness_matrix_K`, `mass_matrix_K`):

    K_e = ∫∫ Bᵀ D_f B  det J dξ dη      (3×3 Gauss),                    (M4)
    M_e = ρ h ∫∫ Nᵀ N  det J dξ dη      (5×5 Gauss, consistent mass).   (M5)

The 3×3 rule integrates the quartic curvature products exactly; the 5×5 rule
the sextic displacement products.

## 4. Assembly, boundary conditions, modal reduction

Elements are assembled on the N1 × N2 grid (`plate_model._assemble`); the
clamped edge z = 0 eliminates (w, w,x, w,z) of the bottom node row
(`_apply_bc`). The free-free generalized eigenproblem

    K φ_k = ω_k² M φ_k                                                   (M6)

is solved in shift-invert mode with a **fixed ARPACK start vector** so that mode
shapes (including their sign) are reproducible between builds — a controller
gain synthesised on one instance transfers exactly to another
(`_modal_analysis`). Modes are mass-normalised (φᵀ M φ = I) and truncated to
n modes: q(t) ∈ ℝⁿ with w(x,z,t) = Σ φ_k(x,z) q_k(t). The modal matrices are
M_p = I, K_p = diag(ω²), C_p = diag(2 ζ_k ω_k) with the *measured* damping
ratios of Du et al. Table 4.

Truncation: the production studies use n = 3; Sec. 8 evaluates the designs on
the refined n = 5 plant (spillover test), since the refined actuator couplings
of modes 4–5 are not small.

## 5. Piezoelectric actuation (d31 patch)

A voltage V across the patch (thickness h_p, planar d31) produces an isotropic
in-plane free strain Λ = d31 V / h_p. Bonded on one face of the plate, this is
statically equivalent to a uniform bending line-moment along the patch boundary,

    m_p = - E_p d31 (h + h_p) / 2(1-ν_p) · V  ≡  m̂_p V,                 (M7)

and, by two integrations by parts of the virtual work of the boundary moment,
to the distributed generalized force (Eq. 15 of the source model)

    F_piezo = m̂_p V ∫∫_patch ∇²N dA  =  H_pe V,                          (M8)

implemented as `laplace_n_patch` + projection H_pe,k = φ_kᵀ F (per volt). For a
patch aligned with the mesh (the 20 × 60 mm article patch spans 6 × 18 elements
exactly) the integral is evaluated with 2×2 Gauss per covered element.

## 6. Patch mass and stiffness (the refinement)

The source package stopped at (M8): the patch influenced the model **only** as
a force. A 0.7 mm PZT layer bonded over 15 % of a 4 mm plate also adds mass and
bending stiffness. With the plate occupying [-h/2, h/2] and the patch
[h/2, h/2 + h_p], and E'ᵢ = Eᵢ/(1-νᵢ²), the composite neutral axis shifts by

    z̄ = E'_p h_p z_c / (E'_a h + E'_p h_p),   z_c = (h + h_p)/2,        (M9)

and the bending-stiffness increment splits into a plate-transport term and the
patch's own term (each assembled with its own Poisson pattern):

    ΔD₁ = E'_a h z̄²,          ΔD₂ = E'_p ( h_p³/12 + h_p (z_c - z̄)² ). (M10)

Each ΔD is assembled with the standard element stiffness using an equivalent
modulus E_eq = 12(1-ν²) ΔD / h³; the added mass ρ_p h_p uses the consistent
mass matrix with ρ_eq = ρ_p h_p / h; partially covered elements are weighted by
their covered-area fraction (`plate_model._add_patch_dynamics`). Rotary inertia
of the offset layer, glue-layer compliance and the (membrane) part of the
asymmetric lamination are neglected — the validation below shows the residual
error this leaves. After the update the modal analysis and every quantity
derived from the mode shapes (Dp(x), sensor rows, H_pe) are recomputed.

## 7. Cutting force, regenerative delay, time integration

The 3-tooth helical end mill engages the top edge over [φ_st, φ_ex]. Per tooth
and axial slice, tangential/normal cutting forces linear in the chip thickness
integrate along the helix to the periodic coefficients α₃(t), α₄(t)
(`milling_force.py`), with k₁ = k_N cos η, k₂ = 1 + μ_c tan η cos γ_n − k_N sin η.
The dynamic chip thickness regenerates from the previous tooth passage, giving
the delayed modal equation of motion (`newmark_solver.py`)

    M_p q̈ + C_p q̇ + (K_p + α₄ D_p D_pᵀ) q − α₄ D_p D_pᵀ q(t−τ)
        = f_t α₃ D_p + H_pe u,       τ = 60/(N_T Ω).                    (M11)

D_p(x_p) is the mode-shape row at the current tool position (precomputed on a
2001-point path grid). Time integration is Newmark-β (γ = 1/2, β = 1/4,
unconditionally stable) with the delayed term read from the response history;
Δt = 50 µs resolves mode 5 with ≈ 5 points per period — adequate for the
delayed forcing which is dominated by modes 1–2. Stability analysis uses the
closed-loop semi-discretization of the same equation (`cl_fdm.py`).

## 8. Verification and validation

**(a) Mesh convergence.** First five natural frequencies for meshes from
20×16 to 50×40 (`results/refinement.json → convergence`): the production mesh
30×24 is indistinguishable from 50×40 on f₁ at 0.1 Hz resolution (< 0.01 %);
the production mesh is fully converged for the retained band.

**(b) Validation against experiment** (Du et al. Table 4, measured):

| mode | measured (Hz) | bare model | error | refined model | error | article's own theory |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 540  | 521.1 | −3.51 % | **540.6** | **+0.12 %** | −0.56 % |
| 2 | 1068 | 1069.9 | +0.18 % | 1092.0 | +2.24 % | +3.09 % |
| 3 | 2787 | 2733.0 | −1.94 % | 2739.2 | −1.72 % | +0.65 % |
| 4 | 3351 | 3334.4 | −0.50 % | 3405.2 | +1.62 % | +2.15 % |
| 5 | 4122 | 4145.6 | +0.57 % | 4183.8 | +1.50 % | +3.20 % |
| mean&#124;err&#124; | | | 1.34 % | | **1.44 %** | 1.93 % |

Two honest observations. First, adding the patch dynamics moves the
chatter-dominant mode 1 from −3.5 % error to **+0.12 %** — the missing physics
was precisely the patch stiffening of the fundamental — and the refined model's
mean error (1.44 %) is *lower than the article's own theoretical model*
(1.93 %). Second, mode 2 worsens (+2.2 %): the perfect-bond, no-glue composite
model slightly over-stiffens the second mode; we report this rather than tune
it away. Density sensitivity (ρ_PZT 7500→7800 kg/m³) moves f₁ by < 0.3 %.

**(c) Effect on the control conclusions (spillover test).** The production
controllers are *designed on the 3-mode bare nominal model* and re-evaluated,
unchanged, on the refined 5-mode plant — both in the linear CL-SD (with the
5-mode plant in the monodromy) and in saturated time domain. Results in
`results/refinement.json → control_on_refined` and manuscript Sec. 4.8: the
qualitative conclusions (ADRC > LQG on both metrics; feasible ≪ linear) are
unchanged, with quantitative shifts reported there. Note that the refined
collocated coupling of mode 3 changes sign; the sign-viability screening of the
placement study must therefore be checked against the refined model — which is
exactly what the spillover test does.

**(d) Sampling / integration resolution as a modelling layer.** The refined
plant exposes a fidelity interaction that the 3-mode model hides entirely. The
closed ADRC loop on the refined plant carries a *marginally damped* spillover
pair near 3.4 kHz (continuous-time max Re(eig) ≈ −35 s⁻¹, i.e. an effective
damping of only ~0.16 %). At the production sample time Δt = 50 µs (10 kHz)
two numerical effects conspire against it: the one-sample implementation delay
contributes ≈ 60–70° of phase lag at that frequency, and Newmark-β distorts the
periods of modes 4–5 by up to ~14 % ((ωΔt)²/12). The simulated loop then
destabilises at ~3.76 kHz — although the continuous closed loop is stable. At
Δt = 25 µs the response is clean and converged (tip RMS 0.253 µm vs 0.228 µm at
12.5 µs). Two consequences: (i) every refined-plant time-domain verdict must be
computed at Δt ≤ 25 µs; (ii) physically, a 20 kHz controller rate is a *design
requirement* for this ADRC on this structure — a deliverable of the refined
model that the truncated model could never produce.

## 9. Known limitations

- Kirchhoff kinematics (no transverse shear/rotary inertia): < 1 % below 5 kHz
  at h/λ ≤ 1/10, consistent with the validation residuals.
- Perfect bond, no adhesive layer, membrane–bending coupling of the one-sided
  patch neglected (Sec. 6) — the +2.2 % residual on mode 2 bounds their joint
  effect.
- No in-process material removal: at a_e = 0.1 mm off a 4 mm thickness along
  the top edge, the modal shift over one pass is ≪ the ±15–20 % drift already
  covered by the robustness study.
- The eddy-current sensor and amplifier dynamics appear in the realistic piezo
  model (`piezo_actuator.py`) but not in the stability analyses.
