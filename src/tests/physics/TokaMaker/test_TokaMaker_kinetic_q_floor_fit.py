#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""TokaMaker tests: kinetic reconstruction, the q floor inside the single-slice fit (``kinetic.fit``): the tests
of ``test_TokaMaker_kinetic_q_floor*.py`` that need the fit module, on the same synthetic geometry and stand-in
forward model (no solver).

Checked: a fit made inside an open branch log appends its assemblies to it; a branch flip at the branch boundary
is recorded on a fit and the record changes nothing; the floor-effect record and the floor records cannot change a
fit (default path and sawtooth path); the opt-in frozen branch keeps the first assembly's branch for the whole fit;
the fit's default free parameter is the correction amplitude, and the fit refuses the sawtooth parameters on the
default core treatment (the two assertions removed from the q-floor files of the kinetic helper package).

Ported from the reconstruction's own tests with every bound unchanged (bodies verbatim, imports re-pointed; the
helpers of the q-floor files imported from them by name).  Not ported (they need the reconstruction's time-dependent
series driver, which is not part of this package): 2 tests of the q-floor file, 7 of the continuous-floor file and
5 of the options file (14 in all).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from OpenFUSIONToolkit.TokaMaker.kinetic import current_model as CM
from OpenFUSIONToolkit.TokaMaker.kinetic import fit as FIT
from OpenFUSIONToolkit.TokaMaker.kinetic import sawtooth as SAW
from OpenFUSIONToolkit.TokaMaker.kinetic.diffusion import CurrentState
import test_TokaMaker_kinetic_fit_unit as FU
from test_TokaMaker_kinetic_q_floor_continuous import GEO, IP_MET, LR, Q_LAND, _fq
import test_TokaMaker_kinetic_q_floor_continuous as _QCC

CENTERS = np.array([0.15, 0.3, 0.45, 0.6])


def _default_fit(monkeypatch, ip=IP_MET, q_floor=1.0, free=("a_corr",), target_shift=0.03):
    """A fit on the DEFAULT core treatment (q floor, resistive, continuous) whose every evaluation is a real
    assembly on the fixture (no forward solve): the residuals are the scored model q at four radii against a target
    (the default model's q plus a shift); Jacobian columns, trials and the noise solve all assemble."""
    lag, kin, src = LR._fixture_inputs()
    kin = FU.SimpleNamespace(**vars(kin), source="toy")
    src = FU.SimpleNamespace(**vars(src), source="toy", convention={})
    cfg = CM.CurrentModelSettings(r0=GEO.R0, q_floor=q_floor)
    a0 = CM.assemble(lag, kin, src, {}, cfg, ip)
    target = np.interp(CENTERS, a0["q_model"]["rho"], a0["q_model"]["q"]) + target_shift
    fm = FU.FakeFM()
    FU._install(monkeypatch, fm)

    def _eval(fm_, params):
        asm = CM.assemble(lag, kin, src, params, cfg, ip)
        fm_.r = (np.interp(CENTERS, asm["q_model"]["rho"], asm["q_model"]["q"]) - target) / 0.02
        return asm

    def consistent(fm_, kin_, src_, params, cfg_, start, z0, label, initial_profiles=None, state_fn=None):
        fm_.n_solves += 1
        asm = _eval(fm_, params)
        solve = FU.SimpleNamespace(stats={"q_0": 1.0}, status="converged", converged=True,
                                   record={"policy": {"verification": {"verdict": "PASS"}}})
        return CM.ConsistentResult(True, "consistent", 1, asm, lag, solve, {"passes": [], "reason": None}, 0.01)

    def lagged(fm_, lag_, kin_, src_, params, cfg_, ip=None, start=None, z0=None, label="", state=None):
        fm_.n_solves += 1
        asm = _eval(fm_, params)
        return FU.SimpleNamespace(converged=True, status="converged"), asm, "converged"

    def residual_vector(fm_, ms, groups, settings, roles):
        return fm_.r.copy(), [("flux_loop", s_) for s_ in "abcd"], FU.SimpleNamespace(model={}, excluded={}, notes=[])

    monkeypatch.setattr(FIT, "_consistent", consistent)
    monkeypatch.setattr(FIT._cm, "lagged_solve", lagged)
    monkeypatch.setattr(FIT, "residual_vector", residual_vector)
    return FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(free=free))


def _assert_same_fit(o, ref):
    assert o.status == ref.status and o.record["n_solves"] == ref.record["n_solves"]
    np.testing.assert_array_equal(o.theta, ref.theta)
    np.testing.assert_array_equal(o.covariance, ref.covariance)
    np.testing.assert_equal(o.chi2, ref.chi2)
    assert o.params == ref.params


def _boundary_level(ip=IP_MET, ds=0.05):
    """A floor level between the plain model axis q at the default bootstrap scale and at the scale one Jacobian
    step above it, so that the center and its s_bs column sit on different branches."""
    lag, _, _ = LR._fixture_inputs()
    rf = CurrentState.uniform_faces(100)
    q = [float(_fq(s_, s_.steady_state(rf))[0]) for s_ in (CM.build_solver(lag, f * lag["jbs"], ip)
                                                          for f in (1.0, 1.0 + ds))]
    return 0.5 * (q[0] + q[1]), abs(q[1] - q[0])


def test_a_fit_inside_an_open_branch_log_appends_its_assemblies_to_it(monkeypatch):
    """A fit run while a branch log is open (a series slice's filter fit) appends its assemblies to that log, marked
    with the fit's label; its own record and its result are unchanged."""
    ref = _default_fit(monkeypatch)
    parent, tok = CM.open_branch_log()
    try:
        parent.append(dict(branch="none", q_floor=1.0, q_axis_plain=None))
        o = _default_fit(monkeypatch)
    finally:
        CM.close_branch_log(tok)
    _assert_same_fit(o, ref)
    assert o.record["floor_branch_record"] == ref.record["floor_branch_record"]
    # the helper's own assembly before the fit is logged directly; the fit's assemblies carry the fit's label
    nested = [e for e in parent if "nested_fit" in e]
    assert len(nested) == ref.record["floor_branch_record"]["n_assemblies"] > 0
    assert all(e["nested_fit"] == "fit" for e in nested) and len(parent) == len(nested) + 2
    assert "nested_fit" not in parent[0] and parent[0]["branch"] == "none"
    assert CM.current_branch_log() is None


def test_branch_flip_is_recorded_on_a_fit_at_the_branch_boundary_and_the_record_is_record_only(monkeypatch):
    """A default-path fit (as above, the bootstrap scale free) whose floor level lies between the plain axis q at
    the center and at its s_bs column: the record shows the flip (a Jacobian with columns on different branches, the
    smallest distance of the plain axis q from the level); the record raising leaves the fit bitwise unchanged."""
    level, dq = _boundary_level()
    assert dq > 0.0
    ref = _default_fit(monkeypatch, q_floor=level, free=("s_bs", "a_corr"))
    r = ref.record["floor_branch_record"]
    assert r["note"].startswith("record only") and r["n_assemblies"] > 0
    assert r["branch_changed"] is True and r["n_jacobians_mixed_branches"] >= 1
    assert r["min_abs_plain_q_axis_minus_floor"] <= 0.5 * dq * (1.0 + 1e-9)
    assert r["final_branch"] == ref.record["core"]["floor_branch"]

    def raising(*a, **k):
        raise RuntimeError("injected")

    for patches in ([(CM, "_log_assembly", raising)], [(CM, "branch_flip_record", raising)],
                    [(FIT, "_note_jacobian", raising)], [(CM, "branch_flip_record", lambda *a, **k: dict(x=1))],
                    [(CM, "_log_assembly", raising), (CM, "branch_flip_record", raising),
                     (FIT, "_note_jacobian", raising), (CM, "open_branch_log", raising)]):
        with monkeypatch.context() as mp:
            for mod, attr, fn in patches:
                mp.setattr(mod, attr, fn)
            o = _default_fit(mp, q_floor=level, free=("s_bs", "a_corr"))
        _assert_same_fit(o, ref)
    # away from the boundary nothing flips
    far = _default_fit(monkeypatch)
    assert far.record["floor_branch_record"]["branch_changed"] is False
    assert far.record["floor_branch_record"]["n_jacobians_mixed_branches"] == 0


def test_floor_effect_is_a_record_that_cannot_change_a_fit(monkeypatch):
    kin, src = FU._kin()
    cfg = CM.CurrentModelSettings(r0=1.0, core_relaxation="periodic_orbit")
    outs, calls = {}, []
    real = CM.floor_effect_record

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    def raising(*a, **k):
        calls.append(1)
        raise RuntimeError("injected")

    def garbage(*a, **k):
        calls.append(1)
        return dict(q_axis_achieved=-99.0, verdict="below")

    for name, fn in (("real", counting), ("raising", raising), ("garbage", garbage)):
        fm = FU.FakeFM()
        FU._install(monkeypatch, fm)
        monkeypatch.setattr(CM, "floor_effect_record", fn)
        n0 = len(calls)
        outs[name] = FIT.reconstruct(fm, kin, src, cfg, FIT.FitSettings(free=FU.SAW_FREE),
                                     params0=dict(log_strength=0.5, f=0.37))
        assert len(calls) == n0 + 1                       # made once, at the end
    ref = outs["real"]
    assert ref.record["floor_effect"]["note"].startswith("record only")
    assert outs["raising"].record["floor_effect"]["computed"] is False
    for name in ("raising", "garbage"):
        o = outs[name]
        assert o.status == ref.status and o.record["n_solves"] == ref.record["n_solves"]
        np.testing.assert_array_equal(o.theta, ref.theta)
        np.testing.assert_array_equal(o.covariance, ref.covariance)
        np.testing.assert_equal(o.chi2, ref.chi2)
        assert o.params == ref.params


def test_floor_records_cannot_change_a_default_path_fit(monkeypatch):
    """The default q-floor path: the floor's effect record made inside every assembly (``_floor_effect``) and the
    end-of-fit record (``floor_effect_record``) raising, or returning anything, leave the fit's status, parameters,
    covariance, chi^2 and solve count bitwise unchanged; a failed record says so."""
    ref = _default_fit(monkeypatch)
    assert ref.record["core"]["floor_branch"] == "axis" and ref.record["core"]["acted"]
    assert ref.record["core"]["floor_effect_computed"] is True
    # the fit's own center lands q_min on the floor (the floored model state, before the corrections)
    assert ref.record["core"]["q_floor_met"] is True and 0.0 <= ref.record["core"]["q_min_after"] - 1.0 <= Q_LAND
    assert len(ref.theta) == 3 and np.any(ref.theta != 0.0)

    def raising(*a, **k):
        raise RuntimeError("injected")

    junk = dict.fromkeys(SAW.FLOOR_EFFECT_KEYS, -99.0)
    for name, fe, fer in (("effect raising", raising, None),
                          ("effect garbage", lambda *a, **k: dict(acted=False, rho_floor=None, q_floor_met=False), None),
                          ("effect junk values", lambda *a, **k: dict(junk, acted=False, rho_floor=0.9), None),
                          ("effect not a dict", lambda *a, **k: 7, None),
                          ("end record raising", None, raising),
                          ("end record garbage", None, lambda *a, **k: dict(q_axis_achieved=-99.0)),
                          ("both raising", raising, raising)):
        with monkeypatch.context() as mp:
            if fe is not None:
                mp.setattr(SAW, "_floor_effect", fe)
            if fer is not None:
                mp.setattr(CM, "floor_effect_record", fer)
            o = _default_fit(mp)
        _assert_same_fit(o, ref)
        c = o.record["core"]
        assert c["acted"] and c["rho_floor"] == ref.record["core"]["rho_floor"], name
        if fe is raising or name in ("effect garbage", "effect not a dict"):
            assert c["floor_effect_computed"] is False and np.isnan(c["d_v_loop_rel"]) and c["floor_effect_reason"]
        if fe is not None and name == "effect junk values":
            assert c["d_v_loop_rel"] == -99.0          # junk values stay in the record's own keys only
        if fer is raising:
            assert o.record["floor_effect"]["computed"] is False


# the options file's ``QC`` is the continuous-floor test module (helpers by name; the fit helpers are defined above)
QC = SimpleNamespace(**{k: getattr(_QCC, k) for k in dir(_QCC) if not k.startswith("__")},
                     CENTERS=CENTERS, _default_fit=_default_fit, _assert_same_fit=_assert_same_fit,
                     _boundary_level=_boundary_level)


def _fit_with(monkeypatch, level, free=("s_bs", "a_corr"), **cfg_kw):
    """:func:`test_q_floor_continuous._default_fit` with extra model settings."""
    real = CM.CurrentModelSettings

    def settings(*a, **k):
        return real(*a, **dict(k, **cfg_kw))
    with monkeypatch.context() as mp:
        mp.setattr(QC.CM, "CurrentModelSettings", settings)
        return QC._default_fit(mp, q_floor=level, free=free)


def test_frozen_branch_keeps_the_first_assemblys_branch_for_the_whole_fit(monkeypatch):
    """At the branch boundary the default fit's record shows a flip; with the frozen branch every assembly of the fit
    is on the first assembly's branch (no change, no mixed Jacobian), recorded as frozen."""
    level, dq = QC._boundary_level()
    ref = QC._default_fit(monkeypatch, q_floor=level, free=("s_bs", "a_corr"))
    assert ref.record["floor_branch_record"]["branch_changed"] is True
    fz = _fit_with(monkeypatch, level, q_floor_branch_freeze="first_assembly")
    r = fz.record["floor_branch_record"]
    assert r["branch_changed"] is False and r["n_jacobians_mixed_branches"] == 0 and r["n_assemblies"] > 0
    c = fz.record["core"]
    assert c["branch_frozen"] is True and c["branch_freeze"].startswith("branch frozen")
    assert fz.record["model_settings"]["q_floor_branch_freeze"] == "first_assembly"
    assert CM._BRANCH_FREEZE.get() is None


def test_fit_default_frees_the_correction_amplitude_only():
    assert FIT.FitSettings().free == ("a_corr", "s_press")


def test_fit_refuses_the_sawtooth_parameters_on_the_default_core_treatment():
    lag, kin, src = LR._fixture_inputs()
    with pytest.raises(ValueError, match="opt-in sawtooth core treatments"):
        FIT.reconstruct(SimpleNamespace(ms=None), kin, src, CM.CurrentModelSettings(r0=GEO.R0),
                        FIT.FitSettings(free=("log_strength", "a_corr")))
