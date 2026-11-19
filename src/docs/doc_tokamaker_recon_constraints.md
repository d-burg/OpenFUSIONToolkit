Equilibrium reconstruction constraints in TokaMaker     {#doc_tokamaker_recon_constraints}
================

[TOC]

The \ref OpenFUSIONToolkit.TokaMaker.reconstruction.reconstruction "reconstruction" class collects constraints
(measurements with errors), writes them to a constraint file (`fit.in` by default) and passes that file to the
Fortran solver, which evaluates every constraint on the current equilibrium in `eval_error()` and fits them in
`reconstruct()`. Both use the same evaluation, so a constraint that can be evaluated can also be fitted.

## Error matrix
`eval_error()` returns one row per constraint, in the order of
\ref OpenFUSIONToolkit.TokaMaker.reconstruction.reconstruction.get_constraints "get_constraints()": flux loops,
Mirnov sensors, saddle loops, pressures, coil currents, plasma current, diamagnetic flux, MSE. The columns are

1. the weighted error, \f$ (v - m) w \f$ for every type except coil currents and \f$ q \f$, which use \f$ (m - v) w \f$;
2. the modeled signal \f$ m \f$;
3. the constraint value \f$ v \f$ as read from the constraint file;
4. the non-axisymmetric correction (zero unless conductor corrections are set up).

The weight is \f$ w = 1/\sigma \f$ for the error \f$ \sigma \f$ given in Python. For a pressure constraint at a
point outside the plasma, column 1 compares the value with a linear extrapolation in \f$ \psi \f$ beyond the plasma
boundary (which steers the fit) instead of the modeled signal of column 2.

## Modeled signals and sign conventions
All quantities use TokaMaker's right-handed \f$ (R, \phi, Z) \f$ frame and its poloidal flux per radian
\f$ \psi \f$ (the flux of `get_psi(False)` and `get_field_eval('PSI')`), with
\f$ B_R = -\frac{1}{R}\frac{\partial \psi}{\partial Z} \f$, \f$ B_Z = \frac{1}{R}\frac{\partial \psi}{\partial R} \f$ and
\f$ B_\phi = F(\psi)/R \f$, \f$ F \f$ from the \f$ F F' \f$ profile everywhere (no test of whether the point lies inside
the plasma).

| Type (file id) | Python | Modeled signal |
|---|---|---|
| Mirnov sensor (1) | `add_Mirnov` | \f$ \mathbf{B} \cdot \mathbf{n} \f$, \f$ \mathbf{n} = (n_R, n_\phi, n_Z) \f$ as given (not normalized) |
| Finite-length Mirnov sensor (14) | `add_Mirnov(..., length, n_avg)` | mean of \f$ \mathbf{B} \cdot \mathbf{n} \f$ over the averaging points (below) |
| Plasma current (2) | `set_Ip` | toroidal plasma current [A] (positive values only) |
| Flux loop (7) | `add_flux_loop` | \f$ 2\pi \psi(R, Z) \f$ [Wb] |
| Diamagnetic flux (8) | `set_DFlux` | \f$ \int_{plasma} (B_\phi - F_0/R)\, dA \f$ [Wb] |
| Pressure (9) | `add_pressure` | \f$ p(\psi(R, Z)) \f$ [Pa] |
| Saddle loop (11) | `add_saddle` | \f$ -(\psi_1 - \psi_2) \Delta\phi \f$ for legs 1 and 2 and toroidal width \f$ \Delta\phi \f$ [Wb] |
| Coil current (12) | `set_coil_currents` | coil current [A], without the virtual vertical stability coil share |
| Coil total current (15) | `set_coil_currents(..., total=True)` | total coil current [A] as returned by `get_coil_currents()` (below) |
| MSE (13) | `add_MSE` | \f$ \tan \gamma \f$ or \f$ \gamma \f$ (below) |

**Flux sign.** Since \f$ B_Z = \frac{1}{R}\frac{\partial \psi}{\partial R} \f$ here, while COCOS 11 has
\f$ B_Z = -\frac{1}{2\pi R}\frac{\partial \psi_{11}}{\partial R} \f$ with \f$ \psi_{11} \f$ per full turn, a flux loop signal in
the COCOS 11 convention is \f$ \psi_{11} = -2\pi\psi \f$: it enters a flux loop constraint with the opposite sign.
Field components, currents and the diamagnetic flux are physical quantities and need no conversion between the two.

**Coil currents.** By default (type 12) the modeled current of a coil set is its own current, without its share of
the virtual vertical stability coil (`set_coil_vsc`). With `total=True` (type 15) it is the total that
`get_coil_currents()` returns and that a coil current measurement sees. The virtual coil amplitude is set by each
equilibrium solve, including the solves inside `reconstruct()` with `fitCoils`, which varies the coil currents
themselves; for a coil that carries a virtual coil share, a total-current constraint therefore does not determine the
split between the coil's own current and that share. In the ITER reconstruction test (Fortran driver, `fitCoils`)
with total-current constraints, the vertical stability coil's own current goes from 0.15 MA to 1.14 MA, against a
virtual coil share of -1.00 MA, with the same fit quality and equilibrium within 2.2E-4 relative, and the fit takes
about 1.8 times as long. Constraints may be given for any subset of the coil sets.

## Finite-length Mirnov sensors
For a signed length \f$ L \f$ and \f$ n_{avg} \f$ points, the averaging points lie at offsets
\f$ s_k = (k - (n_{avg}-1)/2)\, |L| / n_{avg} \f$, \f$ k = 0 \ldots n_{avg}-1 \f$ (the midpoints of \f$ n_{avg} \f$ equal parts of a
segment of length \f$ |L| \f$ centered on the sensor) along the unit vector \f$ \mathbf{t} = (n_R, n_Z)/|(n_R, n_Z)| \f$ for
\f$ L > 0 \f$, or \f$ \mathbf{t} = (-n_Z, n_R)/|(n_R, n_Z)| \f$ (rotated by +90 degrees in the R-Z plane) for \f$ L < 0 \f$.
This is the convention of the EFIT probe length `SMP2`. With `length=None` (the default) the sensor is a point probe,
written as a type 1 record exactly as before.

## MSE
With coefficients \f$ c_0 \ldots c_8 \f$ in the IMAS order and the electric field \f$ (E_R, E_Z) \f$ given by the caller,

\f[ \tan \gamma = \frac{c_0 B_Z + c_1 B_R + c_2 B_\phi + c_3 E_R}{c_4 B_Z + c_5 B_R + c_6 B_\phi + c_7 E_R + c_8 E_Z} \f]

and \f$ \gamma = \arctan(\tan \gamma) \in (-\pi/2, \pi/2) \f$ [rad] for `quantity='angle'`, which jumps by \f$ \pi \f$
where the denominator changes sign. A denominator that is exactly zero gives NaN: `eval_error()` warns, and in
`reconstruct()` the evaluation counts as failed (the same treatment as any NaN constraint value). Coefficients and
electric fields for which the denominator vanishes identically are rejected in Python. The coefficients are defined in
the frame above; any map from a device's orientation (directions of the plasma current and toroidal field, sign
convention of the measured angle) is the caller's statement and belongs in the signs of the coefficients. The EFIT
form \f$ \tan\gamma = (A_1 B_Z + A_5 E_R)/(A_2 B_\phi + A_3 B_R + A_4 B_Z + A_6 E_Z + A_7 E_R) \f$ corresponds to
\f$ c_0 = A_1 \f$, \f$ c_3 = A_5 \f$, \f$ c_4 = A_4 \f$, \f$ c_5 = A_3 \f$, \f$ c_6 = A_2 \f$, \f$ c_7 = A_7 \f$, \f$ c_8 = A_6 \f$,
\f$ c_1 = c_2 = 0 \f$.

## Used flag
Every constraint takes `use` (default `True`). An unused constraint is written with weight \f$ -1/\sigma \f$; the solver
clips weights at zero, so the row is still evaluated (columns 2 and 3) with zero weighted error and does not enter the
fit, and the file reads back with the flag and the error. A weight of zero reads back as unused without an error.
An error, if given, must be positive and finite for every type; a constraint is made unused only with `use=False`.
Unused rows still count in the statistics `reconstruct()` prints (number of constraints, RMS weighted error and the
optional confidence estimate).

## Off-mesh sensors
An evaluation point outside the mesh is moved to the nearest mesh boundary node, not to the nearest point of the
boundary, so the move can be much larger than the distance to the mesh: on the ITER test mesh a point 1 cm outside the
boundary moves 0.13 m (the behavior of `reconstruct()` and `eval_error()` is unchanged). `setup_constraints()` reports
such constraints in
\ref OpenFUSIONToolkit.TokaMaker.reconstruction.reconstruction.projected "projected" (row, number of moved points and
the largest distance moved) and prints a warning. Saddle loop legs off the mesh stop the evaluation with an error.

## Constraint file records
Each record is a blank line, the type id, and the lines below (values separated by spaces; the last line holds the
value and the weight). Real numbers are written with 17 significant digits
(the constant `reconstruction.CON_FILE_REAL_FORMAT`), so the solver uses exactly the values given in Python; files written
earlier with 7 significant digits are read as before.

| Type | Lines after the type id |
|---|---|
| 1 | `R Z phi` / `n_R n_phi n_Z` / `value weight` |
| 2, 8 | `value weight` |
| 7, 9 | `R Z` / `value weight` (pressure at the magnetic axis for `R < 0`) |
| 10 | `type psi_N` / `value weight` |
| 11 | `R1 Z1` / `R2 Z2` / `width` / `value weight` |
| 12, 15 | `coil index (1-based)` / `value weight` |
| 13 | `R Z phi` / `c_0 ... c_8` / `E_R E_Z` / `quantity (1: tan gamma, 2: gamma)` / `value weight` |
| 14 | `R Z phi` / `n_R n_phi n_Z` / `L n_avg` / `value weight` |
