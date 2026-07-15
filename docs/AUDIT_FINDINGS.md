# Audit Findings — Verified Defect Register

> **Historical record.** This register audits the earlier learned-feedforward phase
> of the package (DARC-MPC → PALF-LQG → A-PALF-LQG). In **P4 (2026-07-15)** that
> entire controller family was removed at the author's request and replaced by the
> ESO-ADRC family (see `CONTRIBUTION.md`). The protocol fixes documented below
> (P0–P2) remain in force in the current package; the PALF-specific findings refer
> to code that no longer exists and are kept only as the audit record.

Every *critical* and *major* finding below was independently re-verified by an adversarial
review pass against the exact code and the article text (Du et al. 2024, IJMS 274:109257).
All were **CONFIRMED**. Minor findings were not adversarially re-verified.

Legend: 🔴 critical (blocks submission) · 🟠 major (reviewers will catch it) · 🟡 minor

> ## ✅ P0 resolution status (2026-07-14)
>
> The **integrity-blocking (P0)** findings have been fixed in the committed code. See
> `docs/CONTRIBUTION.md` §6.1–6.2 and `docs/REPRODUCED_RESULTS.md` for the corrected
> numbers. Specifically resolved:
> - **Fabricated ×1.30 "DARC" SLD / "31.1 % effective damping"** — removed from
>   `main_simulation.py`, `gen_article_complete_figures.py`, `gen_SLD_academic_style.py`.
>   PALF now honestly shares the LQG boundary.
> - **Training on the evaluation scenario** — replaced by train-once-on-nominal /
>   freeze / evaluate-held-out in `main_simulation.py`.
> - **Rigged "sub-optimal LQG vs optimal DARC" baseline** — both now use identical
>   grid-searched LQG weights (symmetric comparison).
> - **Misleading name + dead code** — `DARC_MPC_v3_Controller` → `PALF_LQG_Controller`;
>   removed `OnlineRLSAdapter`/`lambda_robust` (dead "adaptive"), removed
>   `pretrain_anti_disturbance` (dead), renamed the "Lyapunov filter" →
>   `CLFVoltageGovernor` with an honest docstring, and made the feedforward a genuine
>   phase-only `u_FF(φ)` map (state channel zeroed at train and deploy).
> - **Irreproducible numbers** — regenerated from the committed code.
>
> ## ✅ P1 resolution status (2026-07-14)
>
> The **methodology (P1)** findings are also fixed now:
> - **k1/k2 cutting constants (findings #1–2)** — corrected to Eq. (3)
>   (k1 = kn/cos η = 0.3174, k2 = 1.1258) and deduplicated into the single source
>   `milling_force.cutting_constants`; all inline copies removed.
> - **Inverse crime / spillover / measurement noise** — the simulated plant now carries
>   5 modes while the controllers are designed on 3 (spillover), and 10 nm measurement
>   noise is injected. One eigensolve feeds both via `PlateModel.truncated_view` /
>   `perturbed_copy` (sign-consistent). A conservative Kalman V=1e-12 is kept for
>   robustness.
> - **Identical actuator clipping** — LQG now clips to ±150 V like PALF (`u_max` param).
>
> Consequences (see `docs/REPRODUCED_RESULTS.md`): forces are ~15–20 % stronger, the
> controlled SLD critical depth is 2.05 mm (was 2.54 mm with wrong forces), and the
> honest held-out gains are S1 +5.1 %, S2 +3.9 %, S3 (ω−8 %) +12.7 %, S4 +4.5 %.
>
> ## ✅ P2 resolution status (2026-07-14)
>
> The **strengthening (P2)** findings are also fixed:
> - **SLD machinery (equivalent-damping surrogate, per-mode decoupling)** — replaced by
>   `fdm_stability.compute_closed_loop_SLD`, a rigorous coupled monodromy with the LQG
>   controller (state feedback + Kalman observer) in the loop. PALF = LQG is now a
>   rigorous consequence of ∂u_FF/∂x̂ = 0, not an assertion.
> - **Eq. (15) piezo coefficient** — implemented in `add_piezo_patch` (C_P0 with P_M);
>   coupling ~16 % weaker than the old simplified constant.
> - **Monte Carlo dead code / survivorship bias** — `run_mc_lqg_vs_palf` runs the frozen
>   held-out controllers over uncertainty and reports divergence explicitly (50/50
>   converged; PALF better 100 %, median +5.05 %). Driver `main_robustness_mc.py`.
> - **FEM frequency gap (finding #3)** — `mesh_convergence.py` shows convergence to
>   <0.1 % and reconciles the ~2.6 % offset vs the article's Chebyshev-Ritz theory.
>
> **Remaining (P3):** experimental validation on a physical plate. Everything above is
> simulation. Minor future refinements: worst-position (vs path-averaged) Dp in the SLD,
> fractional-delay interpolation — both second-order here.


## Physical model fidelity (01_core)

### 1. 🔴 k2 cutting-constant formula wrong: dropped parentheses and wrong angle

Article Eq. (3): k2 = 1 + mu_c*tan(eta)*(cos(gamma_n) - kn*sin(gamma_n)) = 1.1258. Code (main_simulation.py line 143, and identically in 04_figures/gen_article_complete_figures.py lines 171/800, gen_SLD_academic_style.py line 106, run/main_realistic_piezo.py line 112, 03_analysis/uncertainty_analysis.py lines 102-104): k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H) = 0.9861. Two errors: the kn*sin term escaped the mu_c*tan(eta)*() product, and it uses sin(ETA_H) (helix, 35deg) instead of sin(GAMMA_N) (rake, 15deg). Result: -12.4% error in k2, propagating into alpha3, alpha4, all cutting forces, chatter onset, and every SLD critical depth in the package. Systematic across all entry points, so all headline numbers (0.14/2.17/3.05mm) are computed with wrong force constants.

**Verification:** Confirmed exactly as stated. Article Eq. (3) is k2 = 1 + mu_c*tan(eta)*(cos(gamma_n) - kn*sin(gamma_n)) = 1.1258 with eta=35deg (helix), gamma_n=15deg (rake), kn=0.26, mu_c=0.2 (Table 3). The code computes k2 = 1 + MU_C*np.tan(ETA_H)*np.cos(GAMMA_N) - KN*np.sin(ETA_H) = 0.9861 at every entry point (main_simulation, main_realistic_piezo, both figure generators, uncertainty_analysis, plus run/ duplicates), committing both claimed errors: the kn*sin term dropped out of the mu_c*tan(eta)*() product,

**Evidence:** Article: article_text.txt:301-302 (Eq. 3 formula with explicit parentheses), article_text.txt:1119-1134 (Table 3 values 35deg/15deg/925MPa/0.26/0.2). Code: 05_main/main_simulation.py:143 (constants at :54-55, propagation at :152-153); 04_figures/gen_article_complete_figures.py:171,800; 04_figures/gen_SLD_academic_style.py:106 (used at :120,133,147); 05_main/main_realistic_piezo.py:112; 03_analysis

**File:** `05_main/main_simulation.py`

### 2. 🟠 k1 very likely misread: kn*cos(eta) instead of the article's kn/cos(eta)

The article typesets k1 as a fraction (kn over cos eta), i.e. k1 = kn/cos(eta) = 0.3174; the 1/cos(eta) factor is the standard helix chip-flow correction in this model family (article ref [74]). Code uses k1 = KN*np.cos(ETA_H) = 0.2130 everywhere (main_simulation.py line 142 and all other scripts) — a 33% underestimate of k1 that skews the alpha3 (feed forcing) vs alpha4 (regenerative stiffness) balance. Caveat: the extracted PDF text is ambiguous ('k1 = kn / cos eta' rendered on two lines), so this should be confirmed against the typeset PDF or ref [74], but kn*cos(eta) matches no standard form.

**Verification:** Confirmed. Code computes k1 = KN*np.cos(ETA_H) = 0.2130 in every occurrence (12 sites incl. run/ duplicates), with no compensating 1/cos(eta) inside milling_force.py (alpha3/alpha4 use k1 as passed, lines 63-64). The article's k1 is a stacked fraction kn/cos(eta) = 0.26/cos(35 deg) = 0.3174: the two-line extraction "k1 = kn" / "cos eta" matches exactly the extraction pattern of the unambiguous fractions in Eq. (4) ((z2-z1)/2 and RT/(2 tan eta), which the code itself implements as division), wher

**Evidence:** 05_main/main_simulation.py:142,595; 05_main/main_realistic_piezo.py:111; 04_figures/gen_article_complete_figures.py:170,799; 04_figures/gen_SLD_academic_style.py:105; 03_analysis/uncertainty_analysis.py:101; 01_core/milling_force.py:58,63-64 (no hidden 1/cos); article_text.txt lines 301-302 (k1 fraction) vs lines 308-319 (Eq. 4 fractions extracted with identical numerator/denominator line-break pa

**File:** `05_main/main_simulation.py`

### 3. 🟠 Natural frequencies 2.6-3.5% below article Table 4, unreconciled

Mesh-converged FEM frequencies (verified at 15x12, 30x24 and 60x48 meshes: 521.1, 1069.9, 2733.0, 3334.4, 4145.6 Hz) are uniformly 2.55-2.97% below the article's theoretical values (537, 1101, 2805, 3423, 4254 Hz) and mode 1 is 3.5% below the measured 540 Hz. Since the mesh is converged, this is a systematic model discrepancy (the article's penalty-based Chebyshev-Ritz appears stiffer; the article's model may also implicitly include patch stiffening), not a numerical artifact. A 3% frequency error shifts chatter frequencies and SLD lobe positions by ~3% in spindle speed, directly undermining comparisons at the 4900 rpm operating point. The README even quotes 'Mode 1, 521 Hz' without ever noting the Table 4 disagreement. For a defensible publication this must be validated or explained (e.g. rho=2830 vs 2700 for AL6061, patch stiffening, BC modeling).

**Verification:** Confirmed with one refinement. The package's FEM frequencies reproduce exactly as claimed (521.1, 1069.9, 2733.0, 3334.4, 4145.6 Hz at 30x24; mesh-converged, 15x12 and 60x48 agree to <0.1%), sitting uniformly 2.55-2.97% below article Table 4 theoretical values (537/1101/2805/3423/4254 Hz) and mode 1 is 3.51% below the measured 540 Hz; README.md:332 quotes 'Mode 1, 521 Hz' and no file in the package mentions 537/Table 4 or reconciles the gap; the 4900 rpm SLD-shift concern is valid since main_sim

**Evidence:** 01_core/plate_model.py:110-121 (eigensolve); 05_main/main_simulation.py:51-52,56 (rho=2830, E=69e9, nu=0.33, RPM=4900); README.md:93-96,332 ('Mode 1, 521 Hz', no Table 4 note); article_text.txt:1454-1488 (Table 4: measured 540/1068/2787/3351/4122, theoretical 537/1101/2805/3423/4254), article_text.txt:683-695 (AL6061, 100x80x4 mm, 2830 kg/m3, 69 GPa, 0.33), article_text.txt:362-373 (Chebyshev poly

**File:** `01_core/plate_model.py`

### 4. 🟠 Piezo coefficient replaces article Eq. (15) with a different formula while docstring claims Eq. (15)

add_piezo_patch uses m_piezo = -E_Pe*d31*(bp+h_Pa)/(2*(1-nu_Pe)) (plate_model.py line 189) and labels it 'Modele de couplage en flexion (Eq. 15)'. The article's Eq. (14)-(15) prefactor is -C_P0*d31/h_Pa with C_P0 = -(1/6)*((1+mu_Pe)/(1-mu_P))*E_P*b_P^2*P_M/(1+mu_P-(1+mu_Pe)*P_M) and P_M the stiffness-ratio expression — numerically these differ by roughly 10-30% in effective actuator gain (hand evaluation: article-style coefficient ~-0.033 vs code ~-0.040 in comparable units). The spatial part (patch-boundary integrals == patch-area Laplacian integral) IS equivalent, but the gain error rescales all control voltages, making the package's voltage/effort comparisons non-comparable to the article's 25%/11% voltage-reduction claims. Either implement Eq. (15) or honestly document the substitute model and calibrate against the article's Fig. 12(b) voltage-to-displacement FRF.

**Verification:** Confirmed. plate_model.py:189 implements m_piezo = -E_Pe*d31*(bp+h_Pa)/(2*(1-nu_Pe)) while the docstring (lines 185-187) claims it is article Eq. (15). The article's Eq. (14)-(15) prefactor -CP0*d31/hPa depends on plate modulus EP, plate Poisson muP, bP^2 and the stiffness ratio PM, none of which appear in the code formula. With the package's own parameters (identical to article Tables 1-2): article coefficient = -3.339e-2 vs code = -3.986e-2, i.e. the code overstates actuator gain by 19.4% — in

**Evidence:** 01_core/plate_model.py:185-189 (docstring 'Eq. 15' + code formula); run/plate_model.py:189 (duplicate); 01_core/kirchhoff_q4.py:219-269 (laplace_n_patch, 'Eq. 15 article' label at line 224); article_text.txt lines 473-475 (Eq. 14 prefactor -CP0*d31/hPa), 515-534 (Eq. 15 CP0 and PM), 696-704 (Table 2 piezo params), 34 and 1613 (25%/11% voltage reduction); 05_main/main_simulation.py:51-59 (parameter

**File:** `01_core/plate_model.py`

### 5. 🟠 Plant model identical to controller design model: no truncated higher modes, no additive uncertainty (inverse crime)

The simulation 'truth' plant is the same 3-mode modal model handed to the LQG/DARC controllers (newmark_solver.py simulates plate.Mp/Cp/Kp with n_modes=3; main_simulation.py builds controllers on the same object). The article deliberately keeps only 2 modes for control and bounds modes 3-5 with additive uncertainty weights (Eqs. 17-20, Table 4 lists 5 modes with measured damping 0.56% and 0.35% for modes 4-5) precisely because spillover onto unmodeled modes is the failure mode of active chatter control. The package cannot exhibit or test spillover, so the +19.3% RMS and 3.05mm SLD claims are evaluated under idealized full-model-knowledge conditions — a reviewer will flag this immediately. Fix: simulate with >=5 modes (the FEM makes this trivial) while designing controllers on 2-3 modes.

**Verification:** CONFIRMED with one nuance. The plant simulated by NewmarkSimulator is the plate's n_modes=3 modal model (Mp/Cp/Kp), and the LQG/DARC controllers are built from the identical 3-mode matrices, so the controller always has full model-order knowledge of the plant: no truncated higher modes exist anywhere in the package, spillover is structurally impossible to exhibit, and the uncertainty_analysis.py Monte Carlo perturbs only parameters of the same 3 modes. Minor correction: controllers are built on

**Evidence:** 01_core/newmark_solver.py:29,57,71 (plant = plate.Mp/Cp/Kp with n = plate.n_modes); 02_controllers/lqg_controller.py:28-29,35-46 (controller A,B,C from same plate.Mp/Kp/Cp, 2*n_modes states); 05_main/main_simulation.py:61 (N_MODES=3), 74-77 (both plate_d and plate_r built with n_modes=3), 146,156,164 (sim on plate_r, controllers on plate_d), 81-86 (S3 parametric perturbation only), 630-636 (SLD ze

**File:** `01_core/newmark_solver.py`

### 6. 🟠 SLD stability analysis drops inter-modal coupling and position dependence present in the article

fdm_stability.py builds one 2x2 delay-oscillator monodromy matrix per mode using a scalar Dp and takes max(rho) over modes (compute_SLD lines 206-216); main_simulation.py additionally feeds it a Dp AVERAGED over all tool positions (lines 599-603). The article's stability analysis (Fig. 6, full-discretization ref [79]) retains the rank-1 coupled term alpha4*Dp^T*Dp across modes and is evaluated at specific positions (start, 1/4, 1/2) because the lobes vary strongly with position. Decoupling is unconservative near mode-interaction regions and averaging Dp is unconservative at the plate free corner where Dp is maximal. Additionally the 'DARC v3' SLD is not computed from any closed-loop model at all — it multiplies the LQG closed-loop damping by an assumed 1.30 factor (main_simulation.py line 636, comment 'estimation : +20%... zeta*1.5'), so the headline 3.05mm vs 2.17mm SLD claim is partly an assumption, not a result.

**Verification:** Confirmed. The SLD in fdm_stability.py treats each mode as an independent 2x2 delay oscillator with a scalar Dp (Dp2 = Dp_modal**2/m_modal at line 113; per-mode loop and max(rho) over modes at compute_SLD lines 206-216), dropping the rank-1 coupled term alpha4*Dp^T*Dp that the article retains (article eq. context line 576, Fig. 7 off-diagonal elements lines 1135-1140) — even though plate_model.py:158 precomputes np.outer(Dp,Dp) which fdm_stability.py never uses. main_simulation.py:599-603 feeds

**Evidence:** fdm_stability.py:25,113,206-216; plate_model.py:158 (unused DpT_Dp); main_simulation.py:599-603,610,624,630-639,687-701; README.md:328-330; article_text.txt:576,867-871,886-896,1135-1140,1498-1499,1985

**File:** `03_analysis/fdm_stability.py`

### 7. 🟠 README makes false claims about the physics implemented

README.md claims: (a) 'Piezo patch addition with stiffness/mass coupling' — add_piezo_patch adds NO mass or stiffness to the plate, only a modal force vector; (b) milling_force.py is 'Numba-compiled' — there is no numba anywhere; (c) kirchhoff_q4 provides 'consistent and lumped options' — only consistent mass exists; (d) the force model is 'Lehmann-Engin' — it is actually the article's (ref [74], Long/Balachandran-type) helical-engagement kernel model; (e) actuator named 'QDA60-200.7' vs the article's QDA60-20-0.7. Any of these being caught by a reviewer damages credibility of the genuinely correct parts.

**Verification:** All five README misstatements are real: (a) add_piezo_patch (plate_model.py:176-201) computes only the modal force-per-volt vector H_Pe_modal (line 196); global M/K assembly (lines 62-96) never includes piezo mass/stiffness, contradicting README.md:151 'stiffness/mass coupling'. (b) No numba/njit/jit anywhere in any .py file; precompute_alpha_periodic (milling_force.py:68-88) is pure Python/NumPy; numba appears only commented-out in requirements.txt:12-13, contradicting README.md:163. (c) kirchh

**Evidence:** README.md:142,151,161,163,17/98/153; 01_core/plate_model.py:176-201 (esp. 196), 62-96; 01_core/milling_force.py:4,57-64,68-88; 01_core/kirchhoff_q4.py:172-184; 01_core/piezo_actuator.py:4,18; requirements.txt:12-13; article_text.txt lines 306, 621-622, 1975-1976

**File:** `README.md`

### 8. 🟡 Rounded 5-point Gauss weights in mass matrix

mass_matrix_K uses abscissas (-0.906,-0.538,0,...) and weights (0.237,0.479,0.569,...) rounded to 3 decimals; the weights sum to 2.001 instead of 2, adding ~0.05% spurious mass (checked numerically). Effect on frequencies is ~0.03% — negligible against the 3% discrepancy but trivially fixable with numpy.polynomial.legendre.leggauss.

**File:** `01_core/kirchhoff_q4.py`

### 9. 🟡 Delay/period rounding simulates 4878 rpm instead of 4900 rpm

n_per = n_tau = round(tau/dt) = 82 at dt=50us tiles the periodic alpha coefficients and sets the regenerative delay to 4.100ms while the true tooth period at 4900 rpm is 4.0816ms (-0.45%). The simulation is self-consistent (coefficients and delay share the same period) but corresponds to a slightly different spindle speed than claimed; near lobe boundaries 0.45% in speed can flip stability. Use dt an exact divisor of tau or interpolate the delayed state.

**File:** `01_core/milling_force.py`

### 10. 🟡 Contact point at mid axial depth instead of the milled edge

main_simulation.py builds Dp along z_p = HP - ap/2 (line 134) whereas the article's point-contact idealization places the interaction on the milled (upper) edge. For ap=0.3mm on an 80mm plate the mode-shape difference is small (<1%), but for the S2 ap=0.6mm and SLD sweeps to ap=4mm the choice matters more and is undocumented.

**File:** `05_main/main_simulation.py`

### 11. 🟡 Piezo patch orientation is an unverified assumption

The 60x20mm patch is mounted with the 60mm side vertical (x:0-0.020, z:0-0.060 in add_piezo_patch call, main_simulation.py line 80). The article specifies only 'left lower corner' with the position optimized in its ref [66]; if the article's patch lies with 60mm along the plate length, H_Pe components (especially mode 2 coupling) change materially. Should be checked against article Fig. 2.

**File:** `05_main/main_simulation.py`

### 12. 🟡 Helix phase uses absolute plate coordinate, and zero-history delayed state

Two cosmetic-but-confusing choices in milling_force.py/newmark_solver.py: (a) theta(t,j,z) is evaluated with z ~ 0.0797-0.080m (absolute plate height) instead of local axial depth, introducing a constant ~11.2 rad tooth phase offset — harmless (time-origin shift) but disagrees with the article's z1/z2 contact-zone definition and will confuse anyone comparing per-tooth phases; (b) q(t-tau)=0 for t<tau models first-pass entry into virgin material, which is fine, but interacts with the feed-forcing term that is active from t=0 — worth one sentence of documentation.

**File:** `01_core/milling_force.py`


## Controllers (02_controllers)

### 13. 🔴 DARC SLD result is fabricated (zeta x 1.30 multiplier)

main_simulation.py line 636: zeta_DARC_eff = zeta_LQG_sld * 1.30, with comments admitting it is an 'estimation' extrapolated from a 4.7% RMS reduction in S1. The README's headline claims (ap_crit 3.05 mm vs 2.17 mm LQG, '+41%', 'DARC-MPC effective damping 31.1%' = 23.9% x 1.3) all derive from this invented multiplier. It is physically wrong on top of being invented: a phase-locked feedforward at tooth-passing harmonics does not change closed-loop poles or the regenerative-chatter stability boundary — the code's own comments (main_simulation.py lines 509-511, 523-525: 'DARC same closed-loop poles (as base = LQG)', 'FF doesn't change CL poles') concede this. The single most damaging defect for publication.

**Verification:** Confirmed, and understated: zeta_DARC_eff = (np.array(zeta_LQG_sld) * 1.30).tolist() at main_simulation.py:636 is an admitted 'estimation' (comments at 630-635 cite three mutually inconsistent justifications: '+20%', '~zeta * 1.5', and the coded x1.30, plus an S1 RMS figure of 4.7%/LQG=0.532 that contradicts README's own +19.20%/LQG=0.628). All README headline SLD/damping claims (3.05 mm vs 2.17 mm, '+41%', 31.1% = 23.9% x 1.3) derive solely from this multiplier via rho_DARC (637-642) and ap_cri

**Evidence:** 05_main/main_simulation.py:630-636 (zeta_DARC_eff = zeta_LQG_sld * 1.30, 'estimation' comments), :637-642 (rho_DARC), :691-701 (ap_crit_DARC), :509-511/:523/:558 (FF doesn't change CL poles); README.md:326-330 (3.05mm/+41%), :338-339 (23.9% -> 31.1%); 04_figures/gen_article_complete_figures.py:719, :830-837; 04_figures/gen_SLD_academic_style.py:142; 02_controllers/darc_mpc_v3_controller.py:32, :20

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/05_main/main_simulation.py`

### 14. 🔴 Controller is trained on the exact evaluation scenario, including the 'robustness' tests

run_scenario() (main_simulation.py lines 164-174) calls darc.pretrain_iterative_simulation(sim_d, alpha3, alpha4, kp_idx, n_iterations=30) with the SAME alpha3/alpha4/kp_idx/n_per used for the subsequent evaluation, then evaluates on that same condition — the controller is scored on its training data. Worse, in S4 (KT +30%) alpha3/alpha4 are computed from the TRUE perturbed KT, so the 'robust to uncertainty' scenario hands the controller the exact perturbed disturbance to train on. The +19.3% average RMS gain and all robustness claims are therefore circular. There is no held-out condition anywhere.

**Verification:** Confirmed, essentially as stated. run_scenario() computes alpha3/alpha4/kp_idx once (main_simulation.py:150-153), pretrains DARC on them for 30 iterations (lines 170-172), then scores it on the identical arrays (line 174). The pretrain loop simulates with those exact arrays (darc_mpc_v3_controller.py:374-377) and trains the NN with zeroed states (line 422), making it a pure phase-to-voltage lookup (phase = k mod n_per, line 567) of the evaluated periodic disturbance; the 4-period training horizo

**Evidence:** main_simulation.py:123,150-153,163-174; darc_mpc_v3_controller.py:363,374-377,412-424,422,567; lqg_controller.py:81-128; uncertainty_analysis.py:63 (no callers); README.md:316-322; gen_article_complete_figures.py:185-210 (paths under /tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/)

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/05_main/main_simulation.py`

### 15. 🔴 'MPC' and 'Deep' in the method name are false — no receding-horizon optimization, no deep network

Tracing DARC_MPC_v3_Controller.step() (darc_mpc_v3_controller.py lines 540-589): (1) Kalman update, (2) dead-code 'adaptation', (3) u_lqg = -K_lqr @ x_hat clipped, (4) u_ff = NN(x_hat, phase) with phase = 2*pi*(k % n_per)/n_per, (5) sum + clip, (6) safety blend. No prediction horizon, no cost minimized online, no constraint handling — nothing that is MPC. The file's own docstring says the v2 offline MPC solver was abandoned. The 'NN' is a single hidden layer of 16 tanh units, (n_x+2)->16->1, roughly 161 parameters for n_x=6, trained by hand-coded plain SGD — not deep by any standard. Publishing under the name 'Deep Adaptive Robust Control with MPC' misrepresents the method and would be caught immediately.

**Verification:** Confirmed in full. DARC_MPC_v3_Controller.step() (lines 540-589) is exactly Kalman update -> dead-code adaptation (lambda_robust computed at 557/559 and never used; rls.omega_hat never consumed) -> clipped u_lqg = -K_lqr@x_hat (562-563) -> u_ff = NN(x_hat, phase) with phase = 2*pi*(k_step % n_per)/n_per (567-571) -> sum+clip (575-576) -> Lyapunov scalar blend (579-580). No horizon, no online cost minimization, no constrained optimization exists anywhere in the file or in 02_controllers. The docs

**Evidence:** 02_controllers/darc_mpc_v3_controller.py:540-589 (step pipeline), :554-559 (dead lambda_robust), :9-10 and :23-24 (v2 MPC abandoned), :77-138 (16-tanh-unit single-hidden-layer NN, hand-coded SGD at 133-136), :85/:215/:241 (n_hidden=16, ff_max=30); 05_main/main_simulation.py:61 and 05_main/main_realistic_piezo.py:48 (N_MODES=3 -> n_x=6 -> 161 params); README.md:188,199-205 (name expansion + false A

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/02_controllers/darc_mpc_v3_controller.py`

### 16. 🔴 'Adaptive' component is dead code — lambda_robust computed and never used

In step() (darc_mpc_v3_controller.py lines 555-559), lambda_robust = self.rls.update(...) is assigned but never referenced again; rls.omega_hat is never fed back into K_lqr, the observer, or the FF. enable_adaptation=True has zero effect on the control signal. The 'A' in DARC is therefore unsupported by the code. Additionally OnlineRLSAdapter (lines 180-198) is not RLS at all — it is a sign-gradient nudge on a mean prediction error with a hard clip to [0.7, 1.3]*omega_nom.

**Verification:** Confirmed. lambda_robust (step(), lines 555-559) is assigned from self.rls.update() and never referenced again; rls.omega_hat is read/written only inside OnlineRLSAdapter (lines 184-197) and never fed into K_lqr (built once, line 291), the Kalman observer (built once, lines 307-322), or the FF NN, so enable_adaptation=True (used in main_simulation.py:168 and gen_article_complete_figures.py:204) has zero effect on the control signal. OnlineRLSAdapter (lines 180-198) is indeed not RLS: it is a sig

**Evidence:** 02_controllers/darc_mpc_v3_controller.py:555-559 (lambda_robust assigned, never used; grep shows only lines 557/559 package-wide), :180-198 (sign-gradient adapter, clip 0.7-1.3*omega_nom at 194-196), :291 (K_lqr fixed), :313-322 (observer fixed), :562-589 (control path uses neither lambda_robust nor omega_hat); 02_controllers/README.md:38,66,97; README.md:188; 05_main/main_simulation.py:168

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/02_controllers/darc_mpc_v3_controller.py`

### 17. 🔴 'Lyapunov safety filter' is a heuristic clamp, not a Lyapunov argument

LyapunovSafetyFilter (darc_mpc_v3_controller.py lines 144-174) checks Vdot + alpha*V <= 0 with V = x'Px, where (a) P solves A_cl'P + P A_cl = -I for the NOMINAL, delay-free, disturbance-free LQR closed loop (line 247), falling back silently to P = I if the solve fails (line 249); (b) the checked dynamics x_dot = A x + B u OMIT the milling force, the regenerative delay term alpha4*Dp'Dp*(q(t)-q(t-tau)), and the periodic stiffness modulation — i.e., the certificate is evaluated against the wrong plant; (c) it is applied to the estimate x_hat, not the state, with no observer-error argument; (d) when the fallback u_lqg itself violates the condition, beta is clipped to [0,1] and the returned u_safe still violates it — the filter silently passes. No claim of V-decrease for the actual closed loop can be made. The article, by contrast, gives a genuine robust-stability condition (mu(T(jw)) < gamma, Eq. 28) covering the delay force and mode truncation as structured uncertainty.

**Verification:** Confirmed in full. The 'Lyapunov safety filter' checks V_dot + alpha*V <= 0 with V = x'Px where P is the Lyapunov solution for the NOMINAL delay-free LQR closed loop (A_cl = A - B*K_lqr, line 246-247), silently falling back to P = I on any exception (line 249); the checked dynamics x_dot = A x + B u (line 155) omit the milling force ft*a3*Dp, the regenerative delay term a4*Dp'Dp*q(t-tau), and the periodic stiffness modulation K_eff = Kp + a4*Dp'Dp that the actual Newmark plant integrates (newmar

**Evidence:** darc_mpc_v3_controller.py:144-174 (filter class; line 155 nominal x_dot; line 160 margin check; lines 165-174 beta clip and unchecked return), :246-251 (P from nominal A_cl, silent P=I fallback), :271-281 (A,B from Mp/Cp/Kp only), :550 and :579 (filter applied to Kalman x_hat), :583 (violated flag only logged); newmark_solver.py:151-154 (actual plant: K_eff = Kp + a4*Dp'Dp, F = ft*a3*Dp + a4*Dp'Dp

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/02_controllers/darc_mpc_v3_controller.py`

### 18. 🔴 NN state inputs are never trained — deployed network injects an untrained random projection of x_hat

pretrain_iterative_simulation builds every training sample with X_train.append(np.zeros(self.n_x)) (line 422, comment: 'fake etat (zeros, NN s'appuie sur phase)'). Since tanh(0)=0, the backprop gradient d_W1 = outer(d_h, inp) is exactly zero for all state-input columns, so those weights remain at their N(0, 0.1) random initialization. At deployment (step(), line 570) the NN is fed the real x_hat (position scaled by 1e6, velocity by 1e3, both O(1) after tanh), so the output contains an arbitrary, untrained function of the state. The 'state-aware' claim is false; the trained part is purely u_FF(phase) — a periodic lookup — and the state pathway is noise. This also invalidates the README's claim that the NN uses the state estimate meaningfully.

**Verification:** Confirmed. pretrain_iterative_simulation trains the FF NN exclusively on zero state vectors (X_train.append(np.zeros(self.n_x)), line 422, with the 'fake etat' comment at line 421). Because _normalize_input applies tanh to the scaled state (lines 104-110), zero state gives exactly-zero state entries in inp, and backward's d_W1 = np.outer(d_h, inp) (line 130) therefore gives exactly zero gradient to the n_x state-input columns of W1, which remain at their N(0,0.1) initialization (line 95). At dep

**Evidence:** 02_controllers/darc_mpc_v3_controller.py:421-422 (zero-state training samples), :95 (W1 N(0,0.1) init), :104-110 (tanh normalization, state entries 0), :130 (d_W1 = outer(d_h, inp) => zero gradient for state columns), :101-102+106-107 (1e6/1e3 scaling), :570 (step feeds real x_hat), :442 (eval also on zero states), :458-508 (pretrain_anti_disturbance with random states, never called); call sites 0

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/02_controllers/darc_mpc_v3_controller.py`

### 19. 🔴 Baseline comparison is not apples-to-apples; README documents a deliberate 'sub-optimal LQG vs optimal DARC base' scheme

Three fairness problems. (1) README.md line 183 states w_q 'default 1e13 for sub-optimal, 1e14 for optimal' and lines 294-295 say 'Build LQG controller (sub-optimal weights) / Build DARC-MPC controller (optimal LQG base + NN)' — documenting an intentionally weakened baseline in the pipeline that produced the reported +19.31%. (2) In the current main_simulation.py the baseline LQG gets a grid search (lines 157-158) while DARC's internal LQR uses fixed defaults (base_w_q=1e14, base_w_qd=1e8, lines 215, 238) — the two 'LQG' components are not guaranteed identical, so the measured delta is not attributable to the feedforward alone. (3) The grid-search criterion itself is wrong: lqg_controller.py lines 117-121 select the weights whose MOST NEGATIVE eigenvalue real part is largest in magnitude (min Re), i.e., it optimizes the fastest pole; stability margin and damping are governed by the slowest pole (max Re). The 'optimized' baseline is optimized for a meaningless quantity.

**Verification:** Confirmed on all three points, with two corrections that strengthen it: (1) The sub-optimal-LQG(w_q=1e13)-vs-optimal-DARC-base(w_q=1e14) scheme is not merely documented in README.md (lines 183, 294-295, 337-338, 408-411) — it is live in the figure-producing code: gen_article_complete_figures.py:190-201 (comments 'LQG with SUB-OPTIMAL weights... not full grid search' vs 'DARC-MPC uses OPTIMAL LQG base (w_q=1e14)') and gen_SLD_academic_style.py:125-142; 05_main/README.md:51-66 ties this exact asym

**Evidence:** README.md:183,294-295,318-322,337-339,408-411; 05_main/README.md:51-52,61-66,86; 04_figures/gen_article_complete_figures.py:190-201,820; 04_figures/gen_SLD_academic_style.py:125-127,138-142; 05_main/main_simulation.py:156-158,164-169; 02_controllers/darc_mpc_v3_controller.py:213,238; 02_controllers/lqg_controller.py:85-86,117-123 (all under /tmp/claude-0/-home-user-active-vibration-control-plate-d

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/02_controllers/lqg_controller.py`

### 20. 🟠 No measurement noise, no spillover, near-singular Kalman assumptions in the headline comparison

main_simulation.py constructs NewmarkSimulator without a piezo object, so in newmark_solver.py line 96-99 y_obs = y_true exactly (no noise, no delay) and u_actual = u (no saturation/hysteresis). The Kalman filter is designed with V_kal = 1e-12 (essentially noise-free measurement) and W_kal = 1e-6*I that does not model the dominant unmodeled input (the milling force). The plant has exactly the same 3 modes as the controller model, so observation/control spillover — the article's central robustness concern, handled there via additive uncertainty weights covering modes 3+ — is structurally impossible in this simulation and therefore untested. Reviewers of a paper positioned against Du et al. will ask for both.

**Verification:** Confirmed in full. main_simulation.py:146-147 builds NewmarkSimulator without a piezo and lines 160/174 call sim.simulate() without the piezo kwarg, so in newmark_solver.py:95-99 y_obs_now = y_true exactly (no noise/delay) and at lines 138-141 u_actual = u (no saturation/hysteresis). lqg_controller.py:61-66 designs the Kalman filter with V_kal=1e-12 and W_kal=1e-6*I by default (never overridden), and the observer (lines 148-176) has no model of the milling force that drives the true plant (newma

**Evidence:** main_simulation.py:146-147,160,174; newmark_solver.py:41,95-99,138-141,152-154; lqg_controller.py:32,61-66,148-176; main_simulation.py:61,74-77,134-135; article_text.txt:586-639 (p.5); README.md:291-339; main_realistic_piezo.py:55-62,154-158

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/05_main/main_simulation.py`

### 21. 🟠 Asymmetric actuator constraints: DARC clips at +/-150 V internally, LQG output is unclamped

DARC_MPC_v3_Controller.step() clips u_lqg, u_proposed, and u_safe at u_max=150 V (lines 563, 576, 580). LQGController.step() (lines 171-176) applies no limit, and with piezo=None the simulator applies none either. The two controllers face different actuator models within the same comparison; any scenario where LQG exceeds 150 V (fig03 plots a 'Sat. piezo' line at 150 V, implying this is the physical limit) makes the LQG results physically unrealizable while DARC's are constrained — corrupting both the y-metrics and the u-effort metrics in an uncontrolled direction.

**Verification:** Confirmed. DARC_MPC_v3_Controller.step() clips u_lqg, u_proposed, and u_safe at u_max=150 V (default u_max=150.0 at line 223), while LQGController.step() returns u = -K_lqr @ x_hat with no limit, and the NewmarkSimulator applies saturation only via piezo.apply() when a piezo model is passed — main_simulation.py runs both controllers with piezo=None, so LQG's voltage enters the plant and the u-metrics unclamped while DARC is bounded at ±150 V. The package itself declares ±150 V as the physical ac

**Evidence:** 02_controllers/darc_mpc_v3_controller.py:223,231,563,576,580 (u_max=150 clips); 02_controllers/lqg_controller.py:171-176 (no clip); 01_core/newmark_solver.py:40,138-141 (piezo=None default, u_actual=u unclamped); 05_main/main_simulation.py:160,169,174 (LQG and DARC simulated without piezo; DARC given u_max=150), 195-197 (u metrics from unclamped signal), 283-284 (axhline(150, label='Sat. piezo') —

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/02_controllers/lqg_controller.py`

### 22. 🟠 LQG SLD assumes ideal LQR damping in the Floquet computation (ignores observer, saturation, gain realization)

main_simulation.py lines 618-627 extract closed-loop modal (omega, zeta) from the full-state LQR eigenvalues and feed them into compute_SLD as if the plate simply had that damping at every (RPM, ap) up to 4 mm. This ignores observer dynamics (LQG poles include estimator poles), voltage saturation at large ap (control authority is finite; at ap ~ 3-4 mm required voltage far exceeds 150 V), and the fact that the controller was designed for the no-delay nominal plant. The 2.17 mm LQG critical depth (README) inherits all of these optimistic assumptions; only the open-loop SLD is defensible as computed.

**Verification:** Confirmed, and if anything understated. main_simulation.py:618-627 extracts (omega, zeta) from lqg_sld.ev_cl, which lqg_controller.py:116 computes as eigvals(A - B@K) — full-state LQR regulator poles only (Kalman estimator poles from lines 61-75 and the discretized delayed observer of lines 148-176 never enter). These zeta are fed to compute_SLD, whose Floquet matrix (fdm_stability.py:126-127) models a passive oscillator with that damping — no controller, actuator, voltage, or saturation exists

**Evidence:** 05_main/main_simulation.py:589, 608-613, 618-627, 636; 02_controllers/lqg_controller.py:35-50, 61-75, 116-122, 148-176; 03_analysis/fdm_stability.py:126-127, 183-229; 01_core/newmark_solver.py:95, 126-142; 01_core/piezo_actuator.py:27-28; README.md:107, 326-330, 407-412; empirical run: u_max=12.8V @ap=0.3mm, 32.2V @0.6mm, 30794V + divergence @1.5mm, 24629V + divergence @3.0mm

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/05_main/main_simulation.py`

### 23. 🟠 No stability analysis for the combined LQG+FF loop on the true delayed periodic system

The only argument available is informal: the FF is bounded (+/-10 V in main, +/-30 V default) and additive on a 'stable' loop. But the loop's stability is only established for the delay-free nominal LTI model (LQR/Kalman eigenvalues); the actual plant is a time-periodic delay-differential system, and even the LQG baseline's stability on it is verified purely by simulation at one operating point. Interaction of the phase-locked FF with the regenerative delay (both are tau-periodic) is never analyzed — e.g., via Floquet analysis of the full closed loop including the FF as a tau-periodic input, which would in fact formalize why the FF does NOT extend the lobes. The article provides exactly the kind of certificate (mu-synthesis over alpha4 in [0.3, 2.9]x, mode perturbations, additive uncertainty) that this package lacks.

**Verification:** Confirmed, with one refinement and one aggravation. Refinement: LQG's stability on the true delayed-periodic plant is verified by simulation at one spindle speed (RPM=4900) across 4 parameter scenarios plus a Monte Carlo parameter study — not literally a single operating point, but still purely time-domain simulation with no formal analysis. Aggravation: the package's own SLD 'closed-loop' results are not Floquet analyses of any closed loop at all — build_FDM_Phi (fdm_stability.py:88-148) contai

**Evidence:** 02_controllers/darc_mpc_v3_controller.py:42-46,155,215,248-249,561; 02_controllers/lqg_controller.py:35-49,116-121; 03_analysis/fdm_stability.py:25,88-148 (no controller in Phi); 05_main/main_simulation.py:56,119-124,165,616-627,630-642 (esp. 636: zeta_DARC_eff = zeta_LQG_sld*1.30); 01_core/newmark_solver.py:5,145-153 (true plant is tau-delayed periodic DDE); README.md:326-330; article_text.txt:88

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/02_controllers/darc_mpc_v3_controller.py`

### 24. 🟠 ILC training loop has no convergence basis and trains on 4 transient periods at the start of the pass

pretrain_iterative_simulation uses u_target = u_ff + 0.3*(-1e6 * gain_sign * y_k) where K_correction=1e6 is admitted 'heuristique' and gain_sign is derived from a crude single-element DC-gain sign (lines 405-413) — no ILC convergence condition (e.g., contraction of the update operator) is checked, so nothing prevents divergence for other configurations. Training simulates only n_sim_steps = 4*n_per (~4 tooth periods, ~49 ms, line 363) at the start of the feed path, while Dp (plate receptance at the cut) varies substantially along the pass — the frozen FF is then applied to the entire 0.5 s cut. Also a likely off-by-one: history_phase is collected from simulator step k=1 onward but paired with y_sim[k] indexed from 0 (lines 396-403).

**Verification:** Confirmed in substance, with two corrections. (1) CONFIRMED: pretrain_iterative_simulation has no ILC convergence basis — u_target = u_ff + 0.3*(-1e6*gain_sign*y_k) with K_correction=1e6 literally commented 'heuristique' (lines 412-417); gain_sign comes from a single-element product D_obs[0]*B[n_modes] (lines 408-410), not the DC gain -C A^-1 B its own comment cites; the loop runs a fixed n_iterations (30 in main_simulation.py:171) and computes y_rms per iteration (line 381) but never uses it fo

**Evidence:** 02_controllers/darc_mpc_v3_controller.py:363 (n_sim_steps = min(nstep, 4*n_per)), :377 (stop_at_time), :408-419 (gain_dc single-element, K_correction=1e6 'heuristique', eta=0.3, clip), :396-402 (history_phase[k] paired with y_sim[k]), :567-568 (phase appended per step call), :381 (y_rms computed, never used for acceptance); 01_core/newmark_solver.py:88 (loop from k=1), :117-119 (step(k_step=k)), :

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/02_controllers/darc_mpc_v3_controller.py`

### 25. 🟠 README misdescribes the implemented controller

README claims: architecture '3 -> 16 -> 16 -> 1 (sigmoid)' (code: (n_x+2)=8 -> 16 -> 1, tanh); 'Adam optimizer' (code: plain SGD, hand-coded backward()); safety filter 'rejects unsafe u* commands' (code blends and can silently pass violations); 'Deep Adaptive Robust Control with MPC' (none of D, A, R-as-certified, or MPC hold, per the defects above). The S1 code comment (main_simulation.py line 634) records a 4.7% RMS gain while the README table averages +19.31% — the reported numbers are not reproducible from the shipped configuration without clarification of which version produced them.

**Verification:** Confirmed on all points. README (root and 02_controllers/README.md) misdescribes the implemented controller: (1) NN is 8->16->1 with tanh (n_input=n_x+2=8, one hidden layer of 16), not 3->16->16->1 sigmoid; (2) training is hand-coded plain SGD, not Adam (only the lr=0.005 value matches the shipped main_simulation.py); (3) the Lyapunov filter blends toward the saturated-LQG fallback (beta=clip(...,0,1)) and returns the fallback un-rechecked when the fallback itself violates the margin, and silent

**Evidence:** README.md:199 (3->16->16->1 sigmoid claim), README.md:201 ("rejects unsafe u*"), README.md:205 (Adam lr=5e-3), README.md:188+316-330 (DARC-MPC name, +19.31% table, 3.05mm SLD); 02_controllers/darc_mpc_v3_controller.py:89-98 (n_input=n_x+2, W1(16,8), W2(1,16)), :114-116 (tanh forward/saturation), :119-136 (plain SGD backward), :153-174 (blending filter, beta=clip, fallback returned unchecked), :245

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/README.md`

### 26. 🟠 Article's controller (mu-synthesis + delayed PD) is absent, so 'beyond the article' has no article baseline

02_controllers/ contains only lqg_controller.py and darc_mpc_v3_controller.py. Neither the mu-synthesis robust controller nor the delayed PD law u_Pd = KPp*y(t-tau) + KPd*ydot(t-tau) (article Eq. 30) is implemented, so DARC is never compared against the method it claims to surpass. A reviewer of a submission positioned relative to Du et al. (2024) will require at minimum the delayed-PD and preferably the combined robust controller as baselines; LQG alone is a strawman the article itself does not use.

**Verification:** Confirmed. 02_controllers/ contains only lqg_controller.py (pure LQR+Kalman) and darc_mpc_v3_controller.py (u = u_LQG + alpha*NN_FF + Lyapunov safety filter; no delayed term). A package-wide grep for mu-synthesis/DK-iteration/structured-singular-value and for KPp/KPd/delayed-PD terms returns zero matches; all tau/delay hits are the regenerative cutting-force delay or a 50-us sensor delay, not a control law. The article's actual controllers — mu-synthesis robust control (Sec. 3.3) and delayed PD

**Evidence:** 02_controllers/ dir listing (only 2 controllers); darc_mpc_v3_controller.py:540-589 (step: u=u_lqg+u_ff, no delayed feedback); lqg_controller.py:1-58; 02_controllers/README.md:3-4 and 88-92 (LQG-only comparison claims); 05_main/main_simulation.py:155-174 (LQG vs DARC only); article_text.txt:1300-1315 (Eq. 30 delayed PD), :943 (Sec 3.3 mu synthesis), :1278 (Sec 3.4 combined), :1360-1385 (article co

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/02_controllers`

### 27. 🟡 pretrain_anti_disturbance is dead/unused and its 'B-pseudoinverse' target is abandoned

The analytically-motivated pretraining (lines 458-538) is never called by main_simulation.py; the computed self.B_pseudoinv (line 297) is never used anywhere. The docstring's claim that the FF target is '-B^+ . alpha4(phase)' describes code paths that do not execute; the actual target inside the unused function is an ad-hoc scaled -0.5*u_FF_max*alpha4/max|alpha4| with a hand-picked gain. Dead code that contradicts the documented method should be removed before release.

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/02_controllers/darc_mpc_v3_controller.py`

### 28. 🟡 Kalman weight choices (W=1e-6*I, V=1e-12) are arbitrary and identical assumptions are untested against any actual noise

Both controllers hard-code the same covariances (lqg_controller.py lines 61-63; darc lines 307-310) with no justification, no units discussion, and no sensitivity study; since the headline simulation injects zero measurement noise, the near-singular V=1e-12 is never stress-tested. If main_realistic_piezo.py adds noise, the design covariances should be tied to the modeled sensor noise, and the comparison rerun there should be the headline result.

**File:** `/tmp/claude-0/-home-user-active-vibration-control-plate-during-milling/529d917d-6fdd-56b4-abdb-d4680dc949e9/scratchpad/simpkg/article_simulation_package/02_controllers/lqg_controller.py`


## Analysis & results pipeline (03_analysis, 05_main)

### 29. 🔴 DARC-MPC SLD is fabricated by a hardcoded 1.30 damping multiplier, not computed

main_simulation.py lines 630-642: zeta_DARC_eff = zeta_LQG * 1.30, then the open-loop FDM is rerun with this inflated damping and the result is presented as the 'SLD DARC-MPC v3'. The same constant appears in 04_figures/gen_SLD_academic_style.py:142 and gen_article_complete_figures.py:719,837 and in run/main_simulation.py:636. The in-code justification is self-contradictory: the comment says 'estimation: +20% effective damping' but applies 30%, and cites an observed S1 RMS reduction of only 4.7%. The README headline 'a_p crit DARC = 3.05 mm (+41% vs LQG, 21.7x OL)' therefore derives entirely from an arbitrary constant. No Floquet analysis of the NN-in-the-loop system exists, and none is possible without linearizing the NN (e.g., harmonic/describing-function or trajectory linearization) — nothing of the kind is implemented. Floquet theory applies to linear periodic systems; presenting this as an FDM result for a nonlinear NN controller is not defensible in a publication.

**Verification:** Confirmed. The "SLD DARC-MPC v3" is not computed from the controller: 05_main/main_simulation.py:636 hardcodes zeta_DARC_eff = zeta_LQG_sld * 1.30 and reruns the linear FDM (lines 637-642) with this inflated damping; identical code at run/main_simulation.py:636, and the same 1.30 constant at 04_figures/gen_SLD_academic_style.py:142 and gen_article_complete_figures.py:719,837. The in-code justification is self-contradictory (comment line 630 says '+20% effective damping', line 634 cites a 4.7% S1

**Evidence:** 05_main/main_simulation.py:630-642 (multiplier at :636, contradictory comments :630,:634,:635, labeled 'DARC-MPC v3' at :651); run/main_simulation.py:636; 04_figures/gen_SLD_academic_style.py:142; 04_figures/gen_article_complete_figures.py:719,837; README.md:330,338-339; 03_analysis/fdm_stability.py:88-180 (linear per-mode FDM, no controller input); 02_controllers/darc_mpc_v3_controller.py (no lin

**File:** `05_main/main_simulation.py`

### 30. 🔴 README '31.1% effective damping' is the same 1.30 fudge factor, presented as a modal result

README table 'Modal Damping (Mode 1)': DARC-MPC effective 31.1% = 23.9% (LQG optimal closed-loop) x 1.30. gen_article_complete_figures.py:719 computes zeta_DARC_eff = zeta_DARC * 1.30 and plots it as DARC damping. Meanwhile main_simulation.py fig06 (lines 509-535) openly admits 'DARC uses same K_lqr base... FF doesn't change CL poles' and plots identical poles for LQG and DARC. The package simultaneously asserts the NN does not change closed-loop poles and reports a 31.1% modal damping for it.

**Verification:** Confirmed. gen_article_complete_figures.py:719 hardcodes zeta_DARC_eff = zeta_DARC * 1.30 (comment line 718: 'LQG optimal + ~30% from FF') and plots it as the 'DARC-MPC (effective)' modal-damping bar; README.md:338-339 reports exactly this as a result (23.9% x 1.30 = 31.07 -> 31.1%) with no disclaimer. Meanwhile main_simulation.py:509-511 and :523 state 'DARC uses same K_lqr base... NN_FF doesn't change CL poles... poles are same as LQG' and plot identical zeta_LQG bars for both controllers (lin

**Evidence:** 04_figures/gen_article_complete_figures.py:718-719,733-735,830,837; README.md:332-339; 05_main/main_simulation.py:509-511,523-535,576,630-636; 04_figures/gen_SLD_academic_style.py:142

**File:** `04_figures/gen_article_complete_figures.py`

### 31. 🔴 NN feedforward is trained on the evaluation scenario (data leakage / unfair baseline), so the +19.3% RMS gain is not a like-for-like comparison

In main_simulation.py run_scenario (lines 152-174), alpha3/alpha4 are computed with the scenario's ACTUAL parameters (e.g., KT_actual = 1.3x nominal in S4, ap = 0.6 mm in S2), and darc.pretrain_iterative_simulation is run with those same arrays, the same kp_idx and the same simulator time base used for final scoring. In S1 the NN is trained via 30 iterations of closed-loop simulation on the identical plant and trajectory on which it is then evaluated. The LQG baseline is designed once on the nominal model and receives no equivalent adaptation. The claimed 'robustness' gains in S2-S4 therefore partly reflect per-scenario retraining with true perturbed cutting forces, not robustness of a frozen controller. For an honest comparison, either freeze the NN after nominal training for all scenarios, or give LQG an equivalent adaptive mechanism (e.g., ILC-tuned feedforward).

**Verification:** Confirmed with two refinements. Core claim verified: in run_scenario, alpha3/alpha4 are built from the scenario's ACTUAL parameters (S2 ap=0.6mm, S4 KT=1.3x nominal; main_simulation.py:119-124, 152-153, with kt/za_low entering the force arrays in milling_force.py:68-88), and darc.pretrain_iterative_simulation is called with those same alpha3/alpha4/kp_idx arrays and an identical-time-base simulator (main_simulation.py:170-172 vs scoring at line 174; sim_d built at line 147 with same dt/T_end). P

**Evidence:** main_simulation.py:152-153 (alpha3/alpha4 from KT_actual, HP-ap), :170-174 (pretrain with same alpha3/alpha4/kp_idx then score), :90-95 (reset keeps NN weights), :156-160 + lqg_controller.py:105-125 (LQG design force-blind); darc_mpc_v3_controller.py:363, 374-377, 396-437 (30 closed-loop training iterations on actual forces); milling_force.py:68-88; README.md:314-322 (+19.31%). S3 exception: main_

**File:** `05_main/main_simulation.py`

### 32. 🟠 Closed-loop LQG SLD uses an equivalent-damping surrogate, not a closed-loop monodromy matrix

main_simulation.py lines 617-627: extract_modes() takes eigenvalues of A - B@K (LQR full-state feedback, from lqg_controller.py:116) and maps them to per-mode (omega, zeta), then reruns the open-loop per-mode FDM. This omits (a) the Kalman observer dynamics (the simulated controller is output-feedback with a one-step-delayed noisy measurement, not state feedback), (b) the actuator path Hpe and +/-150 V saturation, (c) cross-mode coupling in A - BK (the closed-loop system is not modal-decoupled, yet it is refit to 3 independent SISO oscillators via extract_modes' brittle sort-by-imaginary-part heuristic). A legitimate closed-loop SLD embeds the controller states in the augmented periodic map. As a minimum, the paper must state that the closed-loop lobes are an equivalent-damping approximation and validate it against time-domain divergence tests at boundary points — no such validation exists. The 2.17 mm LQG figure inherits this uncertainty.

**Verification:** CONFIRMED in substance, with one sub-item corrected. Core defect verified: the closed-loop LQG SLD is an equivalent-damping surrogate, not a closed-loop monodromy computation. main_simulation.py:618-621 builds LQGController, calls optimize_weights, then extract_modes(lqg_sld.ev_cl, N_MODES); lqg_controller.py:116 sets ev_cl = eigvals(A - B@K) — pure LQR full-state-feedback spectrum, no Kalman observer dynamics (L_kal from lqg_controller.py:73-75 never enters ev_cl). extract_modes (main_simulatio

**Evidence:** main_simulation.py:617-627 (LQG SLD via extract_modes of ev_cl into open-loop FDM), main_simulation.py:485-495 (extract_modes sort-by-imag at 487), lqg_controller.py:116 (ev_cl=eigvals(A-B@K)), lqg_controller.py:42-43 (B contains H_Pe_modal — corrects claim (b)), lqg_controller.py:171-176 + newmark_solver.py:95-127 (output-feedback observer with one-step-delayed measurement; piezo=None in main_sim

**File:** `05_main/main_simulation.py`

### 33. 🟠 Per-mode decoupled FDM ignores regenerative modal coupling

compute_SLD (fdm_stability.py:183-229) runs an independent 2-state Floquet problem per mode and takes max(rho) over modes. The physical regenerative term a4(t) * Dp Dp^T is a rank-1 full matrix that couples all retained modes (the article's Fig. 7 explicitly shows the off-diagonal DPrT DPr(1,2), (2,1) elements are significant and position-dependent). Mode-coupled regenerative systems can be destabilized in ways the decoupled analysis misses (combination resonances); the decoupled max is not guaranteed conservative. The Newmark simulation DOES include full coupling (a4 * DpT_Dp matrix in newmark_solver.py:151), so the SLD and the time-domain model are structurally different plants — boundary predictions and simulations cannot be expected to agree.

**Verification:** Confirmed as stated. compute_SLD runs an independent 2-state Floquet problem per mode (fdm_stability.py builds a scalar regenerative gain Dp2 = Dp_i^2/m_i, retaining only the diagonal of the rank-1 coupling matrix, and takes max(rho) over modes at line 216), while the Newmark simulation applies the full rank-1 matrix a4*DpT_Dp (built as np.outer(Dp,Dp) in plate_model.py:158) to both the stiffness (newmark_solver.py:151) and delayed-force (line 153) terms. The article's Fig. 7 (page 8) explicitly

**Evidence:** 03_analysis/fdm_stability.py:109-113 (n_state=2, scalar Dp2), :124 (alpha_k = a4*Dp2), :206-216 (per-mode loop, max at 216), :24-25 (decoupled docstring model); 01_core/newmark_solver.py:151 (K_eff = Kp + a4*DpT_Dp_now), :153 (a4*DpT_Dp_now @ q_delay); 01_core/plate_model.py:158 (np.outer rank-1); 05_main/main_simulation.py:61 (N_MODES=3), :599-613 (Dp_avg used for SLD); article_text.txt lines 886

**File:** `03_analysis/fdm_stability.py`

### 34. 🟠 SLD uses path-averaged Dp instead of worst-position analysis

main_simulation.py lines 599-603 average Dp over 41 sampled tool positions and feed the mean into compute_SLD. Mode-shape amplitude at the free corner is much larger than mid-edge, so the averaged Dp underestimates the regenerative gain at the worst position and overestimates the stability limit there. The article instead computes lobes at start/quarter/half positions and takes the lowest curve (Fig. 6, Fig. 13b). All three README critical depths (0.14/2.17/3.05 mm) are position-averaged quantities being implicitly compared with the article's worst-position limits.

**Verification:** Confirmed, and the defect is worse than claimed. main_simulation.py:599-603 averages the SIGNED Dp over 41 positions (kp=0..2000 step 50) and feeds Dp_avg into all three compute_SLD calls (lines 609/622/637); fdm_stability.py:113 squares this scalar (Dp^2/m), so position variation is gone. Running the actual plate model: mode 1 (521 Hz) is uniform along the free upper edge (avg 6.65 vs max 6.71, harmless), but mode 2 (torsion, 1070 Hz) is antisymmetric (Dp = -/+9.93 at the two free corners, ~0 m

**Evidence:** 05_main/main_simulation.py:599-603 (Dp_avg = np.mean over kp range(0,2001,50)), :609-613/:622-627/:637-642 (Dp_avg into compute_SLD); 03_analysis/fdm_stability.py:113 (Dp2 = Dp_modal**2/m_modal, scalar per mode); 01_core/plate_model.py:99-107 (clamped bottom edge only); 04_figures/gen_SLD_academic_style.py:108-112,181-195 and gen_article_complete_figures.py:802-806,812,824,840 (same averaging prod

**File:** `05_main/main_simulation.py`

### 35. 🟠 Monte Carlo module is dead code and README robustness claims do not match it

run_monte_carlo (uncertainty_analysis.py:63) is called by no script in the package (grep confirms only the definition sites). The README claims 'Random sampling of (omega_n, zeta, K_T) within +/-15%, 100 simulation samples, 95% confidence intervals'; the code defaults are 50 samples, +/-5% on kt/kn/mu_c, +/-2% on omega_n, +/-20% on zeta, and no confidence intervals are computed anywhere — envelope_stats returns pointwise nan-percentiles (p05/p95 of trajectories), which are dispersion envelopes, not CIs. The module's own docstring also contradicts its defaults ('+/-2% ... amortissements' vs zeta_pct=0.20). main_simulation.py's docstring promises 'ROBUSTESSE (Monte Carlo)' but Figure 9 is just the 4 deterministic scenarios. DARC is never subjected to Monte Carlo at all, so no statistical robustness comparison between the two controllers exists.

**Verification:** Confirmed in full. run_monte_carlo (03_analysis/uncertainty_analysis.py:63, duplicated at run/uncertainty_analysis.py:63) is never imported or called by any script — grep across the package finds only definition sites; 03_analysis/README.md:54-56 even documents a nonexistent function name (run_uncertainty_analysis). Root README.md:243-246 claims ±15% sampling of (omega_n, zeta, K_T), 100 samples, and 95% confidence intervals, but code defaults are n_samples=50 (line 65), ±5% on kt/kn/mu_c (lines

**Evidence:** 03_analysis/uncertainty_analysis.py:23-35,63-66,179-191; run/uncertainty_analysis.py:63; 03_analysis/README.md:54-56; README.md:243-246; 05_main/main_simulation.py:20,40-45,119-123,224-227,764-809

**File:** `03_analysis/uncertainty_analysis.py`

### 36. 🟠 Survivorship bias in Monte Carlo envelopes

Trajectories that diverge stop early (stop_threshold) and are stored as NaN beyond stop_idx (uncertainty_analysis.py:86-87,139,153); envelope_stats then uses nanmean/nanpercentile (lines 184-191), so the worst (diverged) samples silently drop out of the mean and p95 envelope at later times, making the with-control envelope look better exactly where failures occur. Divergence counts (stop_idx_yes < nstep-1) are returned but never reported as a failure rate.

**Verification:** Confirmed. Diverged trajectories are truncated at stop_threshold (newmark_solver.py:185-190, stop_idx set at :196) and stored NaN beyond stop_idx (uncertainty_analysis.py:86-87 NaN init; :139 and :153 fill only [:stop_idx+1]); envelope_stats (:184-191) uses nanmean/nanstd/nanmin/nanmax/nanpercentile, so diverged (worst) samples silently drop out of the mean/p95 envelope at later times, biasing the with-control envelope optimistic exactly where failures occur; stop_idx_no/stop_idx_yes are returne

**Evidence:** 03_analysis/uncertainty_analysis.py:86-87,139,153,173-174,184-191; 01_core/newmark_solver.py:185-190,196; 03_analysis/README.md:54-56

**File:** `03_analysis/uncertainty_analysis.py`

### 37. 🟠 Method mislabeled FDM; it is zeroth-order semi-discretization, and a4(t) is under-resolved for this highly interrupted cut

build_FDM_Phi applies exact integration of the ODE part with ZOH on the delayed term and on a4(t) — that is the Insperger-Stepan (2004) semi-discretization (which the docstring itself cites), not the Full-Discretization Method of Ding et al. (2010), which additionally interpolates the delayed state. Calling it FDM in README and paper invites reviewer objection. More materially: with ae=0.1 mm on a D=10 mm tool, the in-cut arc is ~11.5 deg of the 120 deg pitch (~10% duty). main_simulation uses m_div=30, so only ~3 samples per period fall in the cut; the impulsive a4(t) is badly resolved and low-immersion milling is precisely the regime where discretization SLD error is largest (and where period-doubling lobes appear). No m_div convergence study exists anywhere in the package (README says m_div=40; main uses 30).

**Verification:** Confirmed. build_FDM_Phi is zeroth-order Insperger-Stepan semi-discretization, not Ding et al. (2010) FDM: exact expm of the frozen ODE (fdm_stability.py:130) with ZOH on a4(t) (sampled once per subinterval at k*dt_int, lines 168-173) and ZOH on a single un-interpolated delayed block (Bd applied only to D[0:2, 2*n_tau:2*(n_tau+1)], line 144). The mislabel is worse than claimed: the docstring (lines 4-5) calls it "Full-Discretization Method (FDM) d'Insperger-Stepan (2002, 2004)" — wrong name AND

**Evidence:** 03_analysis/fdm_stability.py:4-5,19-22,130,134-144,168-173; 05_main/main_simulation.py:53-54,57,593,613,627,642; README.md:219,227,238; 03_analysis/README.md:32,45; 04_figures/gen_article_complete_figures.py:815,827,843; 04_figures/gen_SLD_academic_style.py:121,134,148

**File:** `03_analysis/fdm_stability.py`

### 38. 🟠 Integer-step uninterpolated delay in Newmark solver shifts the effective spindle speed relative to the SLD

newmark_solver.py:28 n_tau = round(tau/dt): at 4900 RPM with dt=5e-5 s, tau/dt = 81.63 -> 82, a 0.45% delay error with no interpolation of the delayed state (line 145-148 takes qm[:, k-n_tau] directly). precompute_alpha_periodic uses the same n_per=82, so the simulation is self-consistent but effectively runs at ~4878 RPM, whereas the FDM/SLD uses the exact tau at 4900 RPM. Near lobe boundaries a 0.5% speed shift moves the stability limit noticeably; time-domain 'validation' of SLD points is therefore comparing slightly different operating points. Fix: linear interpolation between qm[:,k-n_tau] and qm[:,k-n_tau+1] (or choose dt dividing tau). Startup handling (zero delayed state during the first pass, k-n_tau<=0 -> q_delay=0) is standard and fine; treating the delayed term explicitly while keeping the current a4*DpT_Dp implicit in K_eff is also acceptable at this dt.

**Verification:** Confirmed. newmark_solver.py:28 rounds tau/dt to an integer (81.63 -> 82 at 4900 RPM, NT=3, dt=5e-5) and lines 145-148 use the uninterpolated sample qm[:, k-n_tau], so the time-domain simulation effectively runs at tau=4.100 ms (~4878 RPM, 0.45% shift) while fdm_stability.py:159-162 uses the exact tau (dt_int = tau/m_div) at 4900 RPM; main_simulation.py:151 uses the same n_per=82 rounding for the tiled alpha forcing, making the simulation self-consistent at the shifted period exactly as claimed

**Evidence:** 01_core/newmark_solver.py:28 (n_tau = int(np.round(tau/dt))); newmark_solver.py:145-148 (q_delay = qm[:, k-n_tau], no interpolation); 05_main/main_simulation.py:53,56,64,139,141,151 (NT=3, RPM=4900, DT=5e-5, tau=60/(NT*RPM), n_per = int(np.round(tau/DT))); 01_core/milling_force.py:68-88 (tiling at n_per*dt); 03_analysis/fdm_stability.py:159-162 (exact tau, dt_int = tau/m_div); README.md:324-330 (S

**File:** `01_core/newmark_solver.py`

### 39. 🟠 README RMS numbers contradicted by in-code comment; headline table not reproducible from committed code

main_simulation.py:634 comment states 'In S1: y_RMS LQG=0.532, DARC=0.507 -> reduction 4.7%', while the README table for S1 claims LQG 0.628 / DARC 0.507 / +19.20%. Both cannot come from the same run; the NN pretraining is stochastic-free (seeded rng=42) so results should be deterministic. This suggests the README numbers come from a different (uncommitted) configuration or an earlier LQG tuning, and the suspiciously uniform +19.2/19.5/19.2/19.2% pattern across four very different scenarios is itself a red flag that the gain is dominated by the learned periodic forced-vibration cancellation (identical in all scenarios) rather than chatter-relevant dynamics. The numbers must be regenerated by the committed pipeline before publication.

**Verification:** Core claim confirmed: main_simulation.py:634 comment 'In S1: y_RMS LQG=0.532, DARC=0.507 → reduction 4.7%' directly contradicts the README S1 row (LQG 0.628 / DARC 0.507 / +19.20%, README.md:318 and 05_main/README.md:61), the pipeline is deterministic (NN seeded rng=42 at darc_mpc_v3_controller.py:85/93/359/470; simulator rng seeded default_rng(0) at newmark_solver.py:83-84, and no sensor noise path is active in main_simulation.py), and the committed main_simulation.py (lines 156-158: full LQG w

**Evidence:** 05_main/main_simulation.py:634 (comment), :156-158 (LQG grid search), :630-636 (zeta*1.30 vs '~1.5'); README.md:318-322; 05_main/README.md:14, 38-69, 51-52, 61-66, 85-95; 02_controllers/darc_mpc_v3_controller.py:213, 85, 93, 359, 470; 02_controllers/lqg_controller.py:81-123; 04_figures/gen_article_complete_figures.py:191-203; 01_core/newmark_solver.py:83-84; 01_core/piezo_actuator.py:162 (all unde

**File:** `05_main/main_simulation.py`

### 40. 🟡 ap_crit extraction too coarse to support the quoted precision, with silent saturation at grid edge

main_simulation.py:589 uses ap_arr = linspace(0.0001, 4e-3, 25) (0.1625 mm spacing) and RPM_arr with 172 RPM spacing (nearest point to 4900 is 4914); ap_crit is the first grid value with rho>=1 (lines 691-701), and if a column is entirely stable it silently defaults to 4 mm (lines 699-701) — which would be reported as a genuine critical depth. Quoting 0.14/2.17/3.05 mm (two decimals) requires the finer 60x60 grid of gen_SLD_academic_style.py (0.0677 mm spacing: 0.005+2*0.0677=0.140, +32=2.17, +45=3.05 — consistent, so those numbers ARE grid outputs of that script, not hand-typed), but they are then only as trustworthy as the surrogate-damping inputs feeding that grid, and no bisection refinement is done.

**File:** `05_main/main_simulation.py`

### 41. 🟡 Monte Carlo perturbs omega/zeta but not mode shapes, piezo coupling, or Dp; shallow-copy plate shares these arrays

sample_uncertain_params perturbs kt/kn/mu_c/omega_n/zeta only; plate_perturbed = copy.copy(plate_nominal) (line 122) shares Dp arrays, H_Pe_modal, D_obs — physically, frequency shifts from material removal come with mode-shape and coupling changes (the article's Fig. 7 position-varying DPrT DPr and 10% mass/stiffness perturbations). Also alpha3/alpha4 are recomputed per sample (correct) but the delay tau is never perturbed (spindle-speed uncertainty untested). The frozen-nominal-LQG design is the right robustness protocol (no leakage here) — the sampling scope is just narrower than claimed.

**File:** `03_analysis/uncertainty_analysis.py`

### 42. 🟡 main_realistic_piezo reduction percentages compare a truncated diverging run against full controlled runs

reduction_ideal/reduction_real (lines 169-170) divide controlled max|y| (0.5 s) by uncontrolled max|y| from a run stopped at t=0.2 s while diverging exponentially; the 'reduction %' is therefore an artifact of the arbitrary stop time (running to 0.3 s would inflate it further). Report divergence separately and quote reductions only against a stable reference, or use RMS over a common pre-divergence window.

**File:** `05_main/main_realistic_piezo.py`

### 43. 🟡 extract_modes mode-matching heuristic is brittle

extract_modes (main_simulation.py:485-495) sorts closed-loop eigenvalues by imaginary part and assigns the first n_modes to modes 1..3. Aggressive LQR weights (w_q up to 1e16) can produce overdamped real pole pairs or strongly shifted frequencies, silently mis-assigning modes and leaving zeros in omega/zeta arrays (the k < len(ev_pos) guard). Any such misassignment propagates directly into the LQG SLD inputs.

**File:** `05_main/main_simulation.py`

### 44. 🟡 Duplicate diverging copies of analysis scripts in run/ folder

run/main_simulation.py and run/uncertainty_analysis.py duplicate the 03_analysis/05_main versions (same 1.30 factor at run/main_simulation.py:636). Duplicated result-shaping constants in multiple files make it likely that a future fix in one location leaves stale headline numbers generated by the other.

**File:** `run/main_simulation.py`
