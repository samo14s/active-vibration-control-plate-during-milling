# Literal port: `Plate-FEM/Mindlin_plate` (MATLAB) → `mindlin_q8.py` (Python)

This document records the **line-for-line correspondence** between the MATLAB
Reissner–Mindlin plate element of the `Plate-FEM` repository and the Python
element in [`mindlin_q8.py`](mindlin_q8.py). The element formulation is
reproduced verbatim; only the global mesh bookkeeping (node numbering, boundary
conditions, point evaluation, piezo coupling) is added so the element plugs into
the article simulation package.

Theory reference: `Plate-FEM/Mindlin_plate/Mindlin.markdown`.

---

## 1. Function-by-function correspondence

| MATLAB (`.m`) | Python (`mindlin_q8.py`) | Status |
|---|---|---|
| `Shape_function_M(xi,eta)` → `[N, N_row, derN]` | `shape_function_M(xi, eta)` → `(N, N_row, derN)` | **verbatim** |
| `Jacobian_M(nodeCoor,xi,eta)` → `[Jacobian, XYder]` | `jacobian_M(node_coor, xi, eta)` → `(Jac, XYder)` | **verbatim** |
| `Matrix_der_M(nodeCoor,xi,eta)` → `[Bf, Bs]` | `matrix_der_M(node_coor, xi, eta)` → `(Bf, Bs)` | **verbatim** |
| `Stiffness_matrix_M(E,nu,k,lex,ley,h)` → `K` | `stiffness_matrix_M(E, nu, k, lex, ley, h)` | **verbatim** |
| `Mass_matrix_M(rho,lex,ley,h)` → `M` | `mass_matrix_M(rho, lex, ley, h)` | **verbatim** |
| `Thermal_stress_M(E,nu,alp,lex,ley,h,dT)` → `F` | `thermal_stress_M(E, nu, alp, lex, ley, h, dT)` | **verbatim** |
| element connectivity `DofLEe` (in `Mindlin_temporal.m`) | `build_node_map` + `elem_dofs_M` | equivalent numbering |
| kinematic constraints `DOFb` | `clamped_edge_dofs` | equivalent |
| `w = Nr * w(Dof)` interpolation | `shape_at_point_M` | equivalent |
| — (piezo not in Plate-FEM) | `piezo_moment_load_M` | new, moment analogy |

---

## 2. Shape functions (`Shape_function_M.m`)

8-node Serendipity, node order `1..8` = 4 corners then 4 mid-sides:

```
N1 = -(1-ξ)(1-η)(1+ξ+η)/4      N5 = (1-ξ)(1+ξ)(1-η)/2
N2 = -(1+ξ)(1-η)(1-ξ+η)/4      N6 = (1+ξ)(1-η)(1+η)/2
N3 = -(1+ξ)(1+η)(1-ξ-η)/4      N7 = (1-ξ)(1+ξ)(1+η)/2
N4 = -(1-ξ)(1+η)(1+ξ-η)/4      N8 = (1-ξ)(1-η)(1+η)/2
```

`derN(1,:) = ∂N/∂ξ`, `derN(2,:) = ∂N/∂η` — copied term by term.
The `N` matrix is the `3×24` interpolation matrix with `diag([Ni Ni Ni])`
blocks; `N_row` is the `1×8` row of shape-function values.

## 3. Jacobian (`Jacobian_M.m`)

```
J = [[Σ ∂Ni/∂ξ·xi , Σ ∂Ni/∂η·xi],
     [Σ ∂Ni/∂ξ·yi , Σ ∂Ni/∂η·yi]]
XYder = inv(J)ᵀ · derN        # rows: ∂N/∂x , ∂N/∂y
```

## 4. Strain–displacement matrices (`Matrix_der_M.m`)

DOF order per node `[w, θx, θy]` → columns `0::3`, `1::3`, `2::3`.

```
Bf[0, 2::3] =  ∂N/∂x        (θy)          Bs[0, 0::3] = ∂N/∂x   (w)
Bf[1, 1::3] = -∂N/∂y        (θx)          Bs[0, 2::3] = N       (θy)
Bf[2, 1::3] = -∂N/∂x        (θx)          Bs[1, 0::3] = ∂N/∂y   (w)
Bf[2, 2::3] =  ∂N/∂y        (θy)          Bs[1, 1::3] = -N      (θx)
```

(MATLAB 1-based `3:3:24`, `2:3:23`, `1:3:22` map to Python 0-based
`2::3`, `1::3`, `0::3`.)

## 5. Stiffness (`Stiffness_matrix_M.m`)

```
Hf = E·h²/(12(1-ν²)) · [[1,ν,0],[ν,1,0],[0,0,(1-ν)/2]]
Hs = E·κ/(2(1+ν)) · I₂
2×2 Gauss (ξ,η = ±1/√3):  Kf += Bfᵀ Hf Bf detJ ,  Ks += Bsᵀ Hs Bs detJ
K = h·(Kf + Ks)
```

`κ` is the shear-correction factor (`5/6`). The **uniform reduced 2×2**
integration for both bending and shear is exactly as in the MATLAB source
(it mitigates shear locking of the thick-plate element).

## 6. Mass (`Mass_matrix_M.m`)

```
Ie = diag([1, h²/12, h²/12])            # translational + rotary inertia
2×2 Gauss:  M += Nᵀ Ie N detJ
M = ρ·h·M
```

## 7. Thermal load (`Thermal_stress_M.m`)

```
ε_ther = α·ΔT·[1,1,0]ᵀ
2×2 Gauss:  F += Bfᵀ Hf ε_ther detJ
```

Ported verbatim (kept for completeness / thermal studies; the milling
simulation does not use it).

---

## 8. Additions for package integration (not in Plate-FEM)

These are the assembly-level utilities required by `plate_model.py`. They do
**not** change the element physics; they replicate what the MATLAB driver
scripts (`Mindlin_temporal.m`, `Mindlin_test.m`) do inline, using a clean,
topologically-equivalent Serendipity numbering.

- **`build_node_map(N1, N2)`** — nodes live on a `(2N1+1)×(2N2+1)` grid, minus
  the element-centre positions (both indices odd). Total nodes
  `= (2N1+1)(N2+1) + (N1+1)N2`, identical to the MATLAB `ntot`.
- **`elem_dofs_M(I, J, node_id)`** — the 24 global DOFs of element `(I,J)` in
  the local node order `1..8` matching the element matrices.
- **`clamped_edge_dofs(node_id, edge)`** — clamped-edge DOFs
  (`w = θx = θy = 0`).
- **`shape_at_point_M(...)`** — evaluates `w(x,y) = Σ Ni·wi`, i.e. the modal
  observation / tool-position row vector (analogue of `Nr*w(Dof)` in MATLAB).
- **`piezo_moment_load_M(...)`** — consistent nodal load of an applied isotropic
  bending-moment field over the patch, `∫ Bfᵀ[1,1,0]ᵀ dA` (the Mindlin analogue
  of the article's Kirchhoff `∫∇²N dA` coupling).

---

## 9. Numerical validation

| Test | Ported element | Reference |
|---|---|---|
| Partition of unity `Σ Ni = 1`, `Σ ∂Ni = 0` | ✓ (machine precision) | exact |
| Single free element zero-energy modes | 4 (3 rigid-body + 1 hourglass) | expected for uniform-reduced Q8 |
| **CCCC** thin square plate, `λ₁ = ω a²√(ρh/D)` | **35.98** | **35.99** (Leissa) |
| CCCC modes 2–6 | 73.4, 73.4, 108.3, 131.6, 132.2 | 73.4, 73.4, 108.3, 131.6, 132.2 |
| Cantilever AL6061 (article) mode 1 | **519.4 Hz** | ~521 Hz (article) |

The single spurious hourglass mode of the uniform-reduced element is
non-communicable on the assembled clamped mesh — confirmed by the clean,
spurious-free spectra above and by mesh convergence.

Run the checks with:

```bash
cd tests && ./run_tests.sh          # or see tests/README.md
```
