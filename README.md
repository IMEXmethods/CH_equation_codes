# CH_equation_codes
MATLAB codes for the paper and ReadMe

# SI-IMEX-RK schemes for the Cahn–Hilliard equation

MATLAB codes for the paper "Energy-Stable Semi-Implicit IMEX-RK Methods for the
Cahn–Hilliard Equation" (Sebastiano Boscarino).

Schemes: SI-IMEXRK2 (3 stages, 2nd order) and SI-IMEXRK5-3 (5 stages, 3rd order),
with linearized convex splitting stages solved by preconditioned GMRES.

Requirements: MATLAB (R2025b)

## Files

- "search_shifted_imex_rk3_spd.m" — computes the coefficients of SI-IMEXRK5-3 with `fmincon`
  (order conditions + positivity of the symmetric part of the row difference matrix).
- "CheckIMEXrk3" - verify the order conditions.   
- "CH_energyPLOT.m" — 1D CH, periodic BCs (FFT), random initial datum: energy evolution
  and temporal convergence (Figs. 1–2).
- "CH_eq2D.m" — 2D CH, Neumann BCs (DCT), 512×512 modes, random initial datum:
  snapshots, energy evolution, convergence and long-time energy (Figs. 3–5).
- "Test_effic" — 2D test: linearized SI-IMEXRK5-3 vs nonlinear
  convex splitting scheme solved by Newton's method (Table 1).
- "CH_MOBILTY.m" — 1D CH with variable mobility and forcing term, periodic BCs
  (FFT): solution and energy (Fig. 6)
- "CH_MOBILITY_ACC.m" - convergence with ε = 0.3 (Table 2).

Reference solutions are saved to `.mat` files and reloaded if present; delete them
after changing the parameters. The 2D computations at 512×512 may take several hours.


