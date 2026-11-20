#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""TokaMaker tests: kinetic reconstruction, two pure functions of the forward model without the solver (new tests,
not from the reconstruction's own suite; they close gaps found by deliberate one-line bugs that the tests without
the solver did not catch):

(a) ``verify_states``: the verified-convergence check needs BOTH the axis position and every compared scalar within
    their criteria (an axis-only or scalars-only agreement is a FAIL);
(b) ``parse_history``: the residual history is column 4 of the solver's iteration lines (the norm compared with
    ``nl_tol``) and the vertical-stability coil current is column 7 of the last line, on text written in the
    solver's own format ``(A,I4,6ES12.4)``.
"""
from OpenFUSIONToolkit.TokaMaker.kinetic import forward as FW


def test_verify_states_needs_axis_and_scalars():
    a = dict(R_axis=1.7, Z_axis=0.01, **{k: 1.0 + 0.1 * i for i, k in enumerate(FW.VERIFY_KEYS)})
    assert FW.verify_states(a, dict(a))["verdict"] == "PASS"
    assert FW.verify_states(a, dict(a, R_axis=a["R_axis"] + 10.0 * FW.VERIFY_AXIS_M))["verdict"] == "FAIL"
    k0 = FW.VERIFY_KEYS[0]
    assert FW.verify_states(a, dict(a, **{k0: a[k0] * (1.0 + 10.0 * FW.VERIFY_REL)}))["verdict"] == "FAIL"


def _line(i, vals):
    return "  " + f"{i:4d}" + "".join(f"{v:12.4E}" for v in vals)


def test_parse_history_reads_the_residual_and_vsc_columns():
    rows = [(1.0, 2.0, 3.0e-2, 1.70, 0.01, 1.5e3), (1.1, 2.1, 4.0e-4, 1.71, 0.02, 1.6e3),
            (1.2, 2.2, 5.0e-7, 1.72, 0.03, 1.7e3)]
    text = "  Starting non-linear GS solver\n" + "\n".join(_line(i + 1, r) for i, r in enumerate(rows)) + "\n"
    h = FW.parse_history(text)
    assert h["ok"] and h["n"] == 3
    assert h["residuals"] == [3.0e-2, 4.0e-4, 5.0e-7]
    assert h["vsc_A"] == 1.7e3
