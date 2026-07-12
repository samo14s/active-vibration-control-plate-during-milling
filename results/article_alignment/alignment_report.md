# Article-plant alignment report

The `milling_sim` framework was aligned to the plant of this
repository's article package (cantilever AL6061 plate 100 x 80 x
4 mm, 3-tooth D10 end mill, ae = 0.1 mm down-milling, 4900 rpm,
modelling per Nasiri & Moradi MSSP 224 (2025) 112198).

## Model mapping

| Package quantity | Value | milling_sim mapping |
|---|---|---|
| Galerkin modes | [521.1, 1070.0, 2733.0] Hz | y-direction workpiece modes |
| Damping | [0.0031, 0.0017, 0.0027] | zeta0 per mode |
| Modal masses | [1.0, 1.0, 1.0] kg | m_eff = m / Dp^2 (unit-shape convention) |
| Mode shape at tool | [6.629, 0.0, 0.33] | - |
| Regenerative slope c_reg | -2.690e+07 N/m per m ap | scalar NDDE characteristic equation |
| Force coefficients | KT = 925 MPa, k1 = 0.213, k2 = 0.986 | Kt_eff = k2 KT, Kr_eff = k1/k2 |

## Stability cross-validation (one plant, two methods)

The package predicts stability with a time-domain Floquet
full-discretisation method (Insperger-Stepan); this work adds an
independent frequency-domain averaged-coefficient (ZOA-style)
boundary from the characteristic equation
`1 + c_reg ap (1 - e^{-i w tau}) G(i w) = 0`.

* boundary gap in the VALLEYS (safety-critical region): **20.1 %**
* median boundary gap over the full speed range: 81.7 % (p90 178.1 %)

The two methods agree on the lobe POSITIONS and on the low
valleys; the averaged-coefficient method systematically
overestimates the lobe PEAKS of this ae/D = 1 % sliver cut, the
textbook behaviour for highly interrupted cutting where the
time-periodicity of the directional coefficients (and its flip
lobes) matters - which is precisely why the package's Floquet
FDM is the right production method for this plant, and the ZOA
overlay is a cross-CHECK, not a replacement.

## Adaptive spindle-speed layer on this plant

| Configuration | ap_lim @ 4900 rpm | nominal 0.3 mm stable? | best speed | ap_lim @ best |
|---|---|---|---|---|
| Open loop | 0.08 mm | **no - chatter** | 5199 rpm | 2.00 mm |
| Open loop (conservative, package FDM) | 0.07 mm | no | 5297 rpm | 1.30 mm |
| LQG active control | 3.62 mm | yes | 6642 rpm | 4.00 mm |

Reading: at the article's nominal point the open-loop process is
chatter-limited; the package solves this with active piezo
control (LQG / DARC-MPC).  The adaptive process-parameter layer
of the ASME re-run is COMPLEMENTARY: even without piezo hardware,
moving to the best lobe raises the open-loop limit, and under
LQG the same speed selection multiplies the achievable depth
again - the two strategies compose.

LQG closed-loop modes used: [516.7, 1100.3, 2738.4] Hz, zeta [0.2822, 0.1628, 0.0498].
