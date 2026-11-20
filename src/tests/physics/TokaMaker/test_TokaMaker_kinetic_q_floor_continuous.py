#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""TokaMaker tests: kinetic helper package, the resistive q floor with a CONTINUOUS flattening radius (the default
rule of ``sawtooth``), its branches on the model state's q, and the records of what the floor did.

Synthetic geometry as in ``test_TokaMaker_kinetic_q_floor.py`` (no solver).  Checked:

* the radius is found to round-off, so where the floor is met q_min equals it (not one face's step above it), and the
  radius lies between faces;
* a cell cut by the region's edge takes the overlap-weighted conductivity, cells wholly inside the value at the edge,
  cells outside their own; the state is continuous when the edge crosses a face;
* the radius and q0 converge with the grid; the radius and q0 move continuously and monotonically with the level;
* the three branches: (a) q below the floor on axis -- the region from the axis; (b) below it off axis only -- no
  change, recorded; (c) nowhere -- no change; the time-dependent rule takes on every step the single-slice region of
  the step's sources (the same branches), not a region read from the evolving state;
* the records (loop voltage, l_i, current moved outside the radius, achieved q) are what they say;
* the default rule's own properties (axis current lowered, face q = profiles q, meta = record, Ip after a
  time-dependent step, the not-met fallback, q_min ON the floor where met in an assembled state);
* the record of the floor's branch at every assembly (count per branch, Jacobians with columns on different
  branches, the smallest distance of the plain axis q from the floor);
* the extrapolated axis q of the equilibrium is recorded under its own name (not as an achieved axis q), the other
  axis values named cross-checks with their spread; the model's axis q is read back, valid only where the q check
  passed.

Ported from the reconstruction's own tests with every bound unchanged.  Not ported here: the tests that run the
reconstruction's fit or time-dependent series (that the records cannot change a fit or a series, the series' branch
and l_i records, the cached interval geometry, the nested branch logs): they need those modules.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from OpenFUSIONToolkit.TokaMaker.kinetic import coordinates as CO
from OpenFUSIONToolkit.TokaMaker.kinetic import current_model as CM
from OpenFUSIONToolkit.TokaMaker.kinetic import sawtooth as SAW
from OpenFUSIONToolkit.TokaMaker.kinetic.diffusion import CurrentState
from OpenFUSIONToolkit.TokaMaker.kinetic.geometry import MU0, FluxSurfaceGeometry, OuterLayerGeometry


# ---------------------------------------------------------------- the analytic separatrix-like geometry
# (verbatim from the original coordinates/kinetics/series test file; GEO stands for that file)
R0, B0, F0, DPSI, IP, C = 1.7, 2.0, 3.4, 0.18, 1.2e6, 0.995


def F(x):
    return F0 * (1.0 - 0.02 * x)


def g1(x):
    return (1.0 + 0.08 * x) / R0 ** 2


def q_ref(x, q0=1.1):
    return q0 + 2.2 * x ** 2 - 0.5 * np.log(np.maximum(1.0 - x, 1e-300))


def _lag(q0=1.1, sigma_scale=1.0):
    """A lagged-input dict like ``current_model.evaluate_geometry`` returns, on analytic surfaces."""
    xs = np.linspace(np.sqrt(1e-3), np.sqrt(C), 200) ** 2
    xs[-1] = C
    q = q_ref(xs, q0)
    vps = 4 * np.pi ** 2 * q / (F(xs) * g1(xs))
    iref = IP * (1.6 * xs - 0.6 * xs ** 2)
    bp2 = 2 * np.pi * MU0 * iref / vps
    geom = FluxSurfaceGeometry.from_psi_coordinates(xs, 0.0, DPSI, vps, bp2, g1(xs), F(xs), B0=B0, R0=R0, nrho=201,
                                                    psi_n_edge=C)
    xl = 1.0 - (1.0 - C) * np.geomspace(1.0, 1e-2, 40)
    xl[0] = C
    ql = q_ref(xl, q0)
    vl = 4 * np.pi ** 2 * ql / (F(xl) * g1(xl))
    layer = OuterLayerGeometry.from_psi_coordinates(xl, 0.0, DPSI, vl, g1(xl), F(xl),
                                                    avg_grad_psi2_r2=2 * np.pi * MU0 * IP * (1.6 * xl - 0.6 * xl ** 2) / vl,
                                                    fpol_lcfs=F(1.0))
    ge = np.unique(np.concatenate((np.linspace(1e-3, C, 300), xl)))
    sig = sigma_scale * 1.0e8 * (0.02 + 0.98 * np.clip(1.0 - ge ** 2, 0.0, None) ** 1.5)
    to_rho, to_psin = CM._psin_to_rho(np.asarray(geom.meta["psi_n_samples"]), np.asarray(geom.meta["rho_samples"]))
    rmap = CO.rho_map_from_geometry(geom, layer)
    phys = dict(psi_n=ge, rho=rmap.rho_of_psin(ge), q=q_ref(ge, q0), F=F(ge), inv_r2=g1(ge), b2=F(ge) ** 2 * g1(ge))
    return dict(geom=geom, layer=layer, psi_n_edge=C, to_rho=to_rho, to_psin=to_psin, rho_map=rmap, phys=phys,
                sigma=sig, jbs=B0 * 4.0e5 * np.exp(-((ge - 0.975) / 0.02) ** 2))


GEO = SimpleNamespace(IP=IP, R0=R0, DPSI=DPSI, _lag=_lag)


# (verbatim from the original correction-return test file; LR stands for that file)
def _fixture_inputs():
    lag = GEO._lag()
    ph = lag["phys"]
    ph["inv_r"] = np.sqrt(ph["inv_r2"])
    ph["r_avg"] = np.full_like(ph["psi_n"], GEO.R0)
    lag["dpsi"] = GEO.DPSI
    kin = SimpleNamespace(thermal_pressure_at=lambda psi_n=None, rho=None: 2.0e4 * (1.0 - np.asarray(psi_n, float)),
                          coordinate="psi_n")
    src = SimpleNamespace(jdotb_at=lambda ge, **kw: np.zeros_like(np.asarray(ge, float)),
                          p_fast_at=lambda x, rho=None: np.zeros_like(np.asarray(x, float)))
    return lag, kin, src


LR = SimpleNamespace(_fixture_inputs=_fixture_inputs)


IP_MET = 1.86 * GEO.IP            # the floor 1 acts and is met on the fixture (q0 of the plain state ~0.8)
IP_LOW_Q = 2.2 * GEO.IP           # q < 1 out to rho ~0.56 (as in test_q_floor.py)
ULP2 = 4.5e-16                    # two ulps: the diffusion's 1/(1/sigma) round trip through its resistivity hook
CUT_RTOL = 1e-14                  # the cut cell's weighted conductivity through the same round trip (a few ulps)
Q_LAND = 1e-9                     # q_min above the floor where met: the radius is bisected to 1e-14 in rho


def _solver(ip, jni=None):
    lag = GEO._lag()
    return CM.build_solver(lag, lag["jbs"] if jni is None else jni, ip)


def _fq(sol, st):
    return SAW.face_q(st, sol.geometry_at(st.time).phi_b)


def _reversed_shear():
    """Off-axis co-current source: q0 above the floor, an off-axis minimum below it (as in test_q_floor.py)."""
    lag = GEO._lag()
    rho = lag["phys"]["rho"]
    jni = lag["jbs"] + 8.0 * np.max(np.abs(lag["jbs"])) * np.exp(-0.5 * ((rho - 0.5) / 0.08) ** 2)
    return CM.build_solver(lag, jni, 2.3 * GEO.IP)


def test_continuous_radius_lands_q_min_on_the_floor_between_faces():
    sol = _solver(IP_MET)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    assert rec["floor_rule"] == "continuous" and rec["floor_branch"] == "axis" and rec["acted"]
    assert rec["q_floor_met"] and 0.0 <= rec["q_min_after"] - 1.0 <= Q_LAND
    assert np.min(_fq(s2, st)) == rec["q_min_after"] and rec["rho_floor_inner"] == 0.0
    r = rec["rho_floor"]
    assert not np.any(rf == r) and 0.0 < r < 1.0 and rec["n_cells_cut"] == 1
    assert rec["n_cells_flattened"] == int(np.count_nonzero(rf[:-1] < r))
    # Ip is the steady state's own; the boundary gauge is kept
    assert s2.profiles(st)["ip"] == pytest.approx(sol.profiles(st0)["ip"], rel=1e-12)
    assert st.psi_boundary == st0.psi_boundary
    # the grid rule (FUSE's, comparison) snaps the radius to a face and lands above the floor by up to one face's step
    _, rec_g, _ = SAW.resistive_floor_steady(sol, rf, 1.0, rule="grid")
    assert rec_g["rho_floor"] in rf and rec_g["q_min_after"] >= rec["q_min_after"]
    with pytest.raises(ValueError, match="rule"):
        SAW.resistive_floor_steady(sol, rf, 1.0, rule="face")


def test_cut_cell_takes_the_overlap_weighted_conductivity_and_the_state_is_continuous_in_the_radius():
    sol = _solver(IP_LOW_Q)
    rf = CurrentState.uniform_faces(100)
    rc = 0.5 * (rf[1:] + rf[:-1])
    k = 30
    r = rf[k] + 0.3 * (rf[k + 1] - rf[k])            # cuts cell k, 30 % inside
    s2 = SAW.ResistiveFloor(sol, 1.0, rho_inv=r, rho_faces=rf).with_solver()
    sig = s2.coefficients(0.0, rf).sigma_c
    sig0 = sol.coefficients(0.0, rf).sigma_c
    s_r = float(sol.sigma_fn(0.0, np.array([r]))[0])
    np.testing.assert_allclose(sig[:k], s_r, rtol=ULP2, atol=0)                       # wholly inside
    w = (r - rf[k]) / (rf[k + 1] - rf[k])             # 0.3 to round-off
    np.testing.assert_allclose(sig[k], w * s_r + (1.0 - w) * sig0[k], rtol=CUT_RTOL, atol=0)  # cut cell
    np.testing.assert_allclose(sig[k + 1:], sig0[k + 1:], rtol=ULP2, atol=0)           # wholly outside
    assert rc[k] > r                                   # the cut cell's center is outside: weighted, not left alone
    # at a face the weighting is the cell rule exactly (cells inside the face take the value there, the rest their own)
    s_f = SAW.ResistiveFloor(sol, 1.0, rho_inv=rf[k], rho_faces=rf).with_solver()
    sig_f = s_f.coefficients(0.0, rf).sigma_c
    np.testing.assert_allclose(sig_f[:k], float(sol.sigma_fn(0.0, np.array([rf[k]]))[0]), rtol=ULP2, atol=0)
    np.testing.assert_allclose(sig_f[k:], sig0[k:], rtol=ULP2, atol=0)
    # the state is continuous when the edge crosses a face
    d = 1e-9
    lo = SAW.ResistiveFloor(sol, 1.0, rho_inv=rf[k] - d, rho_faces=rf).with_solver().steady_state(rf)
    hi = SAW.ResistiveFloor(sol, 1.0, rho_inv=rf[k] + d, rho_faces=rf).with_solver().steady_state(rf)
    assert np.max(np.abs(hi.psi - lo.psi)) <= 1e-6 * np.ptp(lo.psi)


def test_radius_and_q0_converge_with_the_grid():
    sol = _solver(IP_MET)
    out = {n: SAW.resistive_floor_steady(sol, CurrentState.uniform_faces(n), 1.0)[1] for n in (50, 100, 200, 400)}
    r = {n: out[n]["rho_floor"] for n in out}
    assert abs(r[50] - r[100]) >= abs(r[100] - r[200]) >= abs(r[200] - r[400])
    assert abs(out[100]["q_axis_after"] - out[400]["q_axis_after"]) <= 1e-3
    assert all(out[n]["q_floor_met"] and 0.0 <= out[n]["q_min_after"] - 1.0 <= Q_LAND for n in out)


def test_radius_and_q0_move_continuously_with_the_level():
    """No steps: the radius moves by less than half a cell between levels 0.0025 apart (the grid rule moves by a whole
    face or not at all), and both the radius and q0 rise strictly with the level."""
    sol = _solver(IP_MET)
    rf = CurrentState.uniform_faces(100)
    levels = np.linspace(0.95, 1.05, 41)
    recs = [SAW.resistive_floor_steady(sol, rf, float(lv))[1] for lv in levels]
    r = np.array([x["rho_floor"] for x in recs])
    q0 = np.array([x["q_axis_after"] for x in recs])
    assert np.all(np.diff(r) > 0) and np.all(np.diff(q0) > 0)
    assert np.max(np.diff(r)) <= 0.5 * (rf[1] - rf[0])
    for lv, x in zip(levels, recs):
        assert x["q_floor_met"] and 0.0 <= x["q_min_after"] - lv <= Q_LAND


def test_floor_region_branches_on_the_model_q():
    rf = np.linspace(0.0, 1.0, 6)
    a = SAW.floor_region(rf, np.array([0.8, 0.9, 1.1, 1.3, 1.6, 2.0]), 1.0)
    assert a["branch"] == "axis" and a["rho_inner"] == 0.0 and a["rho_outer"] == pytest.approx(0.3, rel=1e-12)
    b = SAW.floor_region(rf, np.array([1.2, 0.95, 0.9, 1.1, 1.6, 2.0]), 1.0)
    assert b["branch"] == "off_axis" and b["rho_inner"] == pytest.approx(0.2 * 0.2 / 0.25 + 0.0, rel=1e-12)
    assert b["rho_outer"] == pytest.approx(0.4 + 0.2 * 0.1 / 0.2, rel=1e-12)
    c = SAW.floor_region(rf, np.array([1.2, 1.1, 1.05, 1.1, 1.6, 2.0]), 1.0)
    assert c["branch"] == "none" and c["rho_inner"] is None and c["rho_outer"] is None
    with pytest.raises(SAW.QFloorRefused, match="domain edge"):
        SAW.floor_region(rf, np.array([0.8, 0.9, 1.1, 1.3, 1.6, 0.9]), 1.0)
    # a q that is not finite counts as below the floor
    assert SAW.floor_region(rf, np.array([np.nan, 0.9, 1.1, 1.3, 1.6, 2.0]), 1.0)["branch"] == "axis"


def test_branch_a_axis_below_the_floor_flattens_from_the_axis():
    sol = _solver(IP_LOW_Q)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    assert _fq(sol, st0)[0] < 1.0
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    assert rec["floor_branch"] == "axis" and rec["acted"] and rec["rho_floor_inner"] == 0.0
    sig = s2.profiles(st)["conductivity"]
    s_r = float(sol.sigma_fn(0.0, np.array([rec["rho_floor"]]))[0])
    np.testing.assert_allclose(sig[rf[1:] <= rec["rho_floor"]], s_r, rtol=ULP2, atol=0)
    # the time-dependent rule takes the single-slice region of the sources; on this fixture the floor is not met, so
    # that region ends at the plain state's outermost upward crossing (the steady rule's fallback)
    mod = SAW.ResistiveFloor(sol, 1.0)
    info = mod.region_info(st0)
    up = [c["rho"] for c in SAW.floor_crossings(rf, _fq(sol, st0), 1.0) if c["direction"] == "up"][-1]
    assert info["branch"] == "axis" and info["rho_inner"] == 0.0 and info["rho_inv"] == up
    assert rec["q_floor_met"] is False and info["q_floor_met_steady"] is False    # why rho_inv == up holds here
    assert mod.region(st0) == (info["k"], up) and info["k"] == int(np.count_nonzero(rf[:-1] < up))


def test_branch_b_off_axis_minimum_only_leaves_the_state_unchanged_and_records_it():
    sol = _reversed_shear()
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    q = _fq(sol, st0)
    assert q[0] > 1.0 and q.min() < 1.0 < q[-1]
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    assert s2 is sol and rec["floor_branch"] == "off_axis" and rec["acted"] is False
    assert rec["q_floor_met"] is False and rec["rho_floor"] is None and rec["n_cells_flattened"] == 0
    np.testing.assert_array_equal(st.psi, st0.psi)
    assert [c["direction"] for c in rec["rho_floor_crossings"]] == ["down", "up"]
    assert rec["q_axis_after"] == rec["q_axis_before"] == q[0] and rec["q_min_after"] == q.min()
    # FUSE's rule (from the axis, here the grid rule kept for comparison) raises q0 on the same state: what branch (b)
    # avoids
    _, rec_g, _ = SAW.resistive_floor_steady(sol, rf, 1.0, rule="grid")
    assert rec_g["acted"] and rec_g["q_axis_after"] > rec_g["q_axis_before"]
    # time-dependent: no change on a step from this state
    mod = SAW.ResistiveFloor(sol, 1.0)
    info = mod.region_info(st0)
    assert info["branch"] == "off_axis" and info["k"] == 0 and info["rho_inv"] is None
    ref = sol.step(st0, 1e-3).psi
    np.testing.assert_allclose(mod.with_solver().step(st0, 1e-3).psi, ref, rtol=0, atol=1e-12 * np.ptp(ref))


def test_time_dependent_region_is_the_single_slice_region_of_the_sources():
    """On a diffusion step the region is the single-slice region of the solver's sources at the step's time, whatever
    the state: here a floored state whose own q0 is just above the floor and whose minimum is just below it, off axis
    (its own q would say branch (b), no change, and switch the whole region off for that step), and the plain state
    (its own q would put the edge at its outermost crossing) both get the single-slice region; a step with it is a
    step with that region fixed.  Where the sources' steady state is in branch (b) or (c) nothing changes, whatever
    the state (the reversed-shear and q-above-floor cases are in the branch tests)."""
    sol = _solver(IP_MET)
    rf = CurrentState.uniform_faces(100)
    st_ss, rec, _ = SAW.resistive_floor_steady(sol, rf, 1.0)
    assert rec["floor_branch"] == "axis" and rec["acted"]
    r = rec["rho_floor"] - 0.5 * (rf[1] - rf[0])        # half a cell short of the radius that meets the floor
    st = SAW.ResistiveFloor(sol, 1.0, rho_inv=r, rho_faces=rf).with_solver().steady_state(rf)
    q = _fq(sol, st)
    assert q[0] >= 1.0 > q.min() and int(np.argmin(q)) > 0
    st0 = sol.steady_state(rf)
    for s_ in (st, st0, st_ss):
        mod = SAW.ResistiveFloor(sol, 1.0)
        info = mod.region_info(s_)
        assert info["branch"] == "axis" and info["rho_inner"] == 0.0 and info["rho_inv"] == rec["rho_floor"]
        assert info["k"] == rec["n_cells_flattened"] and info["q_axis_model_steady"] == rec["q_axis_before"]
        fixed = SAW.ResistiveFloor(sol, 1.0, rho_inv=rec["rho_floor"], rho_faces=rf).with_solver().step(s_, 1e-3)
        np.testing.assert_array_equal(mod.with_solver().step(s_, 1e-3).psi, fixed.psi)
    assert SAW.ResistiveFloor(sol, 1.0).region_info(st)["state_branch"] == "off_axis"
    assert SAW.ResistiveFloor(sol, 1.0).region_info(st0)["state_branch"] == "axis"
    # the floored steady state is a fixed point of the time-dependent rule: a step from it leaves it in place
    st1 = SAW.ResistiveFloor(sol, 1.0).with_solver().step(st_ss, 1e-3)
    np.testing.assert_allclose(st1.psi - st1.psi_boundary, st_ss.psi - st_ss.psi_boundary, rtol=0,
                               atol=1e-9 * np.ptp(st_ss.psi))


def test_branch_c_q_above_the_floor_everywhere_changes_nothing():
    sol = _solver(GEO.IP)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    assert s2 is sol and rec["floor_branch"] == "none" and rec["acted"] is False and rec["q_floor_met"]
    np.testing.assert_array_equal(st.psi, st0.psi)
    assert rec["d_v_loop_rel"] == 0.0 and rec["d_li3"] == 0.0 and rec["current_moved_outside_over_ip"] == 0.0
    info = SAW.ResistiveFloor(sol, 1.0).region_info(st0)
    assert info["branch"] == "none" and info["k"] == 0 and info["rho_inv"] is None


def test_the_floor_records_its_effect_against_the_same_model_without_it():
    """Lowering the conductivity at fixed Ip raises the loop voltage (Ip = sum of sigma V and the non-inductive
    current); the current taken from inside the radius returns outside it; the broader current lowers l_i.  The
    recorded values are the ones recomputed from the two states."""
    sol = _solver(IP_LOW_Q)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    p0, p1 = sol.profiles(st0), s2.profiles(st)
    assert rec["d_v_loop_rel"] > 0.0 and rec["current_moved_outside_over_ip"] > 0.0 and rec["d_li3"] < 0.0
    assert rec["d_v_loop_rel"] == pytest.approx(st.meta["v_loop"] / st0.meta["v_loop"] - 1.0, rel=1e-12)
    assert rec["d_li3"] == pytest.approx(p1["li3"] - p0["li3"], rel=1e-12)
    moved = (np.interp(rec["rho_floor"], rf, p0["I_enclosed"]) - np.interp(rec["rho_floor"], rf, p1["I_enclosed"]))
    assert rec["current_moved_outside_over_ip"] == pytest.approx(moved / p0["ip"], rel=1e-12)
    assert rec["q_axis_after"] == pytest.approx(_fq(s2, st)[0], rel=1e-15)
    # the assembly's core record carries them, the floor-effect record reads them, with the achieved q of a solve
    core = dict(rec, q_axis_model=1.02, q_min_model=0.99)
    q_check = dict(axis=dict(q0_eq=1.03))
    fe = CM.floor_effect_record(core, q_check, np.array([1.03, 1.0, 0.98, 1.5, np.nan]))
    assert fe["q_axis_extrapolated_eq"] == 1.03 and fe["q_min_traced_eq"] == 0.98 and fe["floor_branch"] == "axis"
    assert fe["d_v_loop_rel"] == rec["d_v_loop_rel"] and fe["current_moved_outside_over_ip"] == rec[
        "current_moved_outside_over_ip"] and fe["note"].startswith("record only")
    assert CM.floor_effect_record(None)["q_axis_extrapolated_eq"] is None


# ====================================================================== the default rule's own properties
@pytest.mark.parametrize("ip", [IP_MET, IP_LOW_Q])
def test_default_rule_lowers_the_axis_current_and_its_face_q_and_meta_are_consistent(ip):
    """On the default (continuous) rule, where it acts (met: IP_MET; not met: IP_LOW_Q): the core current on the axis
    is lowered and q raised; the operator's face q is the profiles' q; the state's meta says what the record says."""
    sol = _solver(ip)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    p0, p1 = sol.profiles(st0), s2.profiles(st)
    assert rec["floor_rule"] == "continuous" and rec["acted"] and rec["floor_branch"] == "axis"
    assert p1["jdotb"][0] < p0["jdotb"][0] and rec["q_min_after"] > rec["q_min_before"]
    np.testing.assert_allclose(_fq(s2, st), p1["q"], rtol=1e-12)
    m = st.meta["q_floor"]
    assert m["rho_floor"] == rec["rho_floor"] and m["rho_floor_inner"] == rec["rho_floor_inner"] == 0.0
    assert m["form"] == "resistive" and m["rule"] == "continuous" and m["branch"] == "axis" and m["acted"] is True
    assert m["q_floor"] == 1.0 and m["q_floor_met"] is rec["q_floor_met"] is (ip == IP_MET)
    assert rec["floor_effect_computed"] is True


@pytest.mark.parametrize("ip", [IP_MET, IP_LOW_Q])
def test_default_time_dependent_step_conserves_ip(ip):
    sol = _solver(ip)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    mod = SAW.ResistiveFloor(sol, 1.0)
    s2 = mod.with_solver()
    assert mod.rule == "continuous" and mod.region_info(st0)["k"] > 0
    st1 = s2.step(st0, 1e-3)
    assert s2.profiles(st1)["ip"] == pytest.approx(ip, rel=1e-12)


def test_default_rule_not_met_keeps_the_outermost_crossing_with_q_min_below_the_floor():
    sol = _solver(IP_LOW_Q)
    rf = CurrentState.uniform_faces(100)
    st0 = sol.steady_state(rf)
    q0 = _fq(sol, st0)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    up = [c["rho"] for c in SAW.floor_crossings(rf, q0, 1.0) if c["direction"] == "up"][-1]
    assert rec["q_floor_met"] is False and rec["q_min_after"] < 1.0 and np.min(_fq(s2, st)) < 1.0
    assert rec["rho_floor"] == up and rec["floor_branch"] == "axis"


def test_assembled_default_state_lands_q_min_on_the_floor_where_met():
    """The single-slice default (assembly) on the fixture: where the floor is met the floored model state's q_min IS
    the floor (to the bisection's round-off), and the scored model q (no corrections) carries the same minimum."""
    lag, kin, src = LR._fixture_inputs()
    for qf in (0.95, 1.0):                 # the plain model axis q here is ~0.905
        a = CM.assemble(lag, kin, src, {}, CM.CurrentModelSettings(r0=GEO.R0, q_floor=qf), IP_MET)
        c = a["core"]
        assert c["acted"] and c["q_floor_met"] is True and c["floor_branch"] == "axis"
        assert 0.0 <= c["q_min_after"] - qf <= Q_LAND
        assert c["q_min_model"] == pytest.approx(c["q_min_after"], rel=1e-12)


# ====================================================================== branch-flip record (record only)
def test_branch_flip_record_counts_branches_jacobians_and_the_distance_from_the_floor():
    log = [dict(branch="axis", q_floor=1.0, q_axis_plain=0.998), dict(branch="axis", q_floor=1.0, q_axis_plain=0.999),
           dict(branch="off_axis", q_floor=1.0, q_axis_plain=1.0004), dict(branch="axis", q_floor=1.0, q_axis_plain=0.9)]
    jac = [dict(label="it0", branches=["axis", "off_axis"], mixed=True), dict(label="it1", branches=["axis"],
                                                                                mixed=False)]
    r = CM.branch_flip_record(log, jac, dict(floor_branch="axis"))
    assert r["n_assemblies"] == 4 and r["assemblies_per_branch"] == {"axis": 3, "off_axis": 1}
    assert r["n_branch_changes_in_call_order"] == 2 and r["branch_changed"] is True
    assert r["n_jacobians"] == 2 and r["n_jacobians_mixed_branches"] == 1 and r["jacobians_mixed_branches"][0][
        "label"] == "it0"
    assert r["min_abs_plain_q_axis_minus_floor"] == pytest.approx(4e-4, rel=1e-9) and r["final_branch"] == "axis"
    assert r["note"].startswith("record only")
    one = CM.branch_flip_record(log[:2], [], dict(floor_branch="axis"))
    assert one["branch_changed"] is False and one["n_branch_changes_in_call_order"] == 0
    # a series interval entry: the diffusion steps per branch
    s = CM.branch_flip_record([dict(branch="axis", q_floor=1.0, q_axis_plain=None,
                                    branch_steps={"axis": 12, "off_axis": 8}, q_axis_plain_min_abs_dev=2e-3)])
    assert s["branch_changed"] is True and s["diffusion_steps_per_branch"] == {"axis": 12, "off_axis": 8}
    assert s["min_abs_plain_q_axis_minus_floor"] == 2e-3


def _synthetic_axis_values():
    """Axis values for the record tests, all from the synthetic fixture (no discharge): the floored model state's own
    axis face q, its extrapolation from the q check's window, and round values for the equilibrium-side inputs."""
    sol = _solver(IP_MET)
    rf = CurrentState.uniform_faces(100)
    st, rec, s2 = SAW.resistive_floor_steady(sol, rf, 1.0)
    q = _fq(s2, st)
    q_ax = float(q[0])
    q_ext_model = float(CO.axis_q(rf, q, (0.05, 0.15), 2)["q0"])
    return q_ax, q_ext_model


def test_floor_effect_record_names_the_extrapolated_axis_q_and_keeps_the_others_as_cross_checks():
    """The q check's extrapolation of the solved equilibrium is recorded under its own name (not as an achieved axis
    q), with its definition; the trace of the innermost surface, the solver's q_0 (q at psi_N = 0.02) and the model's
    values are named cross-checks with their spread and the estimator's offset on the model, where the model's own
    axis value is known.  The old key names are gone.  Numbers from the synthetic fixture."""
    q_ax, q_ext_model = _synthetic_axis_values()
    q_ext_eq, q_trace, q_s0 = 1.1, 0.5, 1.25
    core = dict(q_floor=1.0, floor_branch="axis", acted=True, q_axis_model=q_ax, q_min_model=q_ax,
                floor_effect_computed=False, floor_effect_reason="RuntimeError: injected")
    q_check = dict(axis=dict(q0_eq=q_ext_eq, q0_model=q_ext_model, fit_rms_rel_eq=0.01, window=[0.05, 0.15], degree=2))
    fe = CM.floor_effect_record(core, q_check, np.array([1.1, 1.05, 1.3]), q_trace, -q_s0)
    assert fe["q_axis_extrapolated_eq"] == q_ext_eq and fe[
        "q_axis_extrapolated_eq_definition"] == CM.Q_AXIS_EXTRAPOLATED_EQ_DEFINITION
    assert fe["q_axis_extrapolated_eq_window"] == [0.05, 0.15] and fe["q_axis_extrapolated_eq_degree"] == 2
    cc = fe["q_axis_cross_checks"]
    assert cc == dict(q_traced_innermost=q_trace, q_solver_q0=q_s0, q_model_axis_extrapolated=q_ext_model,
                      q_model_axis=q_ax)
    assert fe["q_axis_spread"] == pytest.approx(q_s0 - q_trace, rel=1e-12)
    assert fe["q_axis_extrapolation_offset_model"] == pytest.approx(q_ext_model - q_ax, rel=1e-12)
    assert fe["effect_computed"] is False and fe["effect_reason"].startswith("RuntimeError")
    none = CM.floor_effect_record(core)
    assert none["q_axis_extrapolated_eq"] is None and none["q_axis_spread"] is None
    # the new names (the record no longer calls the extrapolation achieved; the old names are removed)
    assert fe["q_axis_extrapolated_eq"] == q_ext_eq and fe["q_min_traced_eq"] == 1.05
    assert fe["q_axis_extrapolated_eq_definition"] == CM.Q_AXIS_EXTRAPOLATED_EQ_DEFINITION
    assert "not the achieved axis q" in CM.Q_AXIS_EXTRAPOLATED_EQ_DEFINITION
    assert not any(k in fe for k in ("q_axis_achieved", "q_axis_achieved_definition", "q_axis_achieved_window",
                                     "q_axis_achieved_degree", "q_axis_achieved_fit_rms_rel", "q_min_achieved",
                                     "deprecated_keys"))
    assert "edge" in CM.floor_effect_record.__doc__ and "not resolved there" not in CM.floor_effect_record.__doc__


def test_floor_effect_record_reads_back_the_model_axis_q_only_where_the_q_check_passed():
    """The read-back of the core level is the model's own axis q of the consistent state; it is marked valid only
    where the q consistency check passed, and invalid (with the reason) without a check, with a failed one, or without
    a model axis q.  Record only."""
    q_ax, _ = _synthetic_axis_values()
    core = dict(q_floor=1.0, q_axis_model=q_ax)
    ok = CM.floor_effect_record(core, dict(passed=True, max_rel=0.004, axis={}))
    assert ok["q_axis_model_readback"] == q_ax and ok["q_axis_model_readback_valid"] is True
    assert ok["q_axis_model_readback_invalid_reason"] is None
    assert ok["q_axis_model_readback_definition"] == CM.Q_AXIS_MODEL_READBACK_DEFINITION
    bad = CM.floor_effect_record(core, dict(passed=False, max_rel=0.02, axis={}))
    assert bad["q_axis_model_readback"] == q_ax and bad["q_axis_model_readback_valid"] is False
    assert "not passed" in bad["q_axis_model_readback_invalid_reason"]
    none = CM.floor_effect_record(core)
    assert none["q_axis_model_readback_valid"] is False and none["q_axis_model_readback_invalid_reason"] == "no q check"
    nomodel = CM.floor_effect_record(dict(q_floor=1.0), dict(passed=True, axis={}))
    assert nomodel["q_axis_model_readback"] is None and nomodel["q_axis_model_readback_valid"] is False
    # a truthy value that is not True does not count as passed
    odd = CM.floor_effect_record(core, dict(passed=1, axis={}))
    assert odd["q_axis_model_readback_valid"] is False


def test_branch_flip_record_states_the_total_and_the_truncation_of_mixed_jacobians():
    """The list of Jacobians with columns on different branches keeps the first 20; the record states the total and
    whether the list was truncated."""
    log = [dict(branch="axis", q_floor=1.0, q_axis_plain=0.99)]
    many = [dict(label=f"it{i}", branches=["axis", "off_axis"], mixed=True) for i in range(25)]
    r = CM.branch_flip_record(log, many + [dict(label="x", branches=["axis"], mixed=False)])
    assert r["n_jacobians_mixed_branches"] == 25 and r["jacobians_mixed_branches_total"] == 25
    assert len(r["jacobians_mixed_branches"]) == 20 and r["jacobians_mixed_branches_truncated"] is True
    few = CM.branch_flip_record(log, many[:2])
    assert few["jacobians_mixed_branches_total"] == 2 and few["jacobians_mixed_branches_truncated"] is False
