#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""TokaMaker tests: kinetic helper package, numerical properties of ``kinetic.diffusion``.

Time-convergence order of each scheme (TR-BDF2 and Crank-Nicolson: 2,
backward Euler: 1) with time-varying Ip, conductivity and non-inductive
current; exact current conservation under the Ip condition; steady state ==
long-time limit; state serialization round trips; sawtooth hooks.  (The
moving-boundary (dPhi_b/dt) term with finite conductivity is tested in
``test_TokaMaker_kinetic_diffusion_guards.py``.)  Synthetic geometry and profiles only.
"""
import numpy as np
import pytest

from OpenFUSIONToolkit.TokaMaker.kinetic.diffusion import CurrentDiffusion, CurrentState, IpBoundary, VloopBoundary
from OpenFUSIONToolkit.TokaMaker.kinetic.geometry import FluxSurfaceGeometry

GEO = FluxSurfaceGeometry.circular(0.6, 1.7, 2.0, n=301, fpol=lambda x: 3.4 * (1.0 - 0.02 * x ** 2))
T_END = 0.4


def sigma_t(t, rho):
    return (4.0e6 * (1.0 - 0.9 * rho ** 2) ** 1.5 + 1.0e5) * (1.0 + 0.3 * np.sin(2 * np.pi * t / T_END))


def jni_t(t, rho):
    return 2.0e5 * (1.0 + t / T_END) * np.exp(-((rho - 0.6) / 0.15) ** 2)


def ip_t(t):
    return 1.0e6 * (1.0 + 0.5 * t / T_END + 0.1 * np.sin(5.0 * t / T_END))


def _solver(bc=None, **kw):
    return CurrentDiffusion(GEO, sigma_t, bc or IpBoundary(ip_t), j_ni=jni_t, **kw)


def _start(sol, n=40):
    return sol.steady_state(CurrentState.uniform_faces(n), t=0.0)


@pytest.mark.parametrize("scheme,theta,order", [("trbdf2", None, 2.0), ("theta", 0.5, 2.0), ("theta", 1.0, 1.0)])
@pytest.mark.parametrize("bc", ["ip", "vloop"])
def test_time_convergence_order(scheme, theta, order, bc):
    boundary = IpBoundary(ip_t) if bc == "ip" else VloopBoundary(lambda t: 1.5 + 0.5 * np.sin(4 * t / T_END))
    sol = _solver(boundary)
    st0 = _start(sol)
    kw = dict(scheme=scheme, theta=theta if theta is not None else 0.5)
    # reference: the semi-discrete solution, approximated by 5120 TR-BDF2 steps for every scheme
    # (a 5120-step backward-Euler reference would itself carry ~6 % of the finest-step error)
    ref = sol.advance(st0, T_END, nsteps=5120, scheme="trbdf2")
    jref = sol.profiles(ref)["jdotb"]
    errs, jerrs = [], []
    for ns in (40, 80, 160, 320):
        s = sol.advance(st0, T_END, nsteps=ns, **kw)
        errs.append(max(np.max(np.abs(s.psi - ref.psi)), abs(s.psi_boundary - ref.psi_boundary)))
        jerrs.append(np.max(np.abs(sol.profiles(s)["jdotb"] - jref)))
    p = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    # theory: p = 2 (TR-BDF2, CN), p = 1 (backward Euler); bound +-0.1, fixed before running
    assert order - 0.1 <= p[-1] <= order + 0.1, (errs, p)
    # <J.B> (a second difference of psi) must converge at the same order: added after an independent review
    # (psi alone cannot reveal undamped edge <J.B> oscillations).  On this smooth case CN passes; on a steep
    # edge or a sampled initial state CN's <J.B> does not converge (module docstring), which is why TR-BDF2
    # is the default.
    pj = np.log2(np.array(jerrs[:-1]) / np.array(jerrs[1:]))
    assert order - 0.1 <= pj[-1] <= order + 0.1, (jerrs, pj)


def test_ip_boundary_conserves_total_current():
    sol = _solver()
    s = _start(sol)
    for k in range(1, 11):
        s = sol.advance(s, k * T_END / 10, nsteps=7)
        p = sol.profiles(s)
        assert abs(p["ip"] / ip_t(s.time) - 1.0) < 1e-13
        # volume integral of the current density: int <J_phi/R> V'/(2 pi) drho = Ip (telescoping, round-off)
        total = np.sum(p["jt_over_r"] * GEO.evaluate(p["rho_cells"], ["dvol_drho"])["dvol_drho"]
                       * np.diff(p["rho_faces"])) / (2 * np.pi)
        assert abs(total / ip_t(s.time) - 1.0) < 1e-12


@pytest.mark.parametrize("scheme,theta", [("trbdf2", 0.5), ("theta", 1.0)])
def test_steady_state_is_long_time_limit(scheme, theta):
    sol = CurrentDiffusion(GEO, lambda t, r: 2.0e6 * (1 - 0.9 * r ** 2) + 1.0e5, IpBoundary(1.2e6),
                           j_ni=lambda t, r: 1.5e5 * np.exp(-((r - 0.6) / 0.15) ** 2))
    rf = CurrentState.uniform_faces(60)
    ss = sol.steady_state(rf)
    # start far from relaxed: flat q = 3 (hollow-ish current)
    st = CurrentState.from_q(GEO, np.linspace(0, 1, 30), 3.0 + 0.0 * np.linspace(0, 1, 30), rf)
    tau = 4e-7 * np.pi * 2.1e6 * 0.6 ** 2      # ~ core resistive time [s]
    out = sol.advance(st, 60 * tau, dt=0.1 * tau, scheme=scheme, theta=theta)
    a, b = sol.profiles(out), sol.profiles(ss)
    np.testing.assert_allclose(a["q"], b["q"], rtol=1e-9)
    np.testing.assert_allclose(a["v_loop"], ss.meta["v_loop"], rtol=1e-9)
    # psi up to its (time-growing) offset
    np.testing.assert_allclose(out.psi - out.psi_boundary, ss.psi - ss.psi_boundary, rtol=1e-9, atol=0)


def test_vloop_steady_state_holds_current():
    sol_ip = CurrentDiffusion(GEO, lambda t, r: 2.0e6 * (1 - 0.9 * r ** 2) + 1.0e5, IpBoundary(1.0e6))
    ss = sol_ip.steady_state(CurrentState.uniform_faces(50))
    sol_v = CurrentDiffusion(GEO, lambda t, r: 2.0e6 * (1 - 0.9 * r ** 2) + 1.0e5, VloopBoundary(ss.meta["v_loop"]))
    out = sol_v.advance(ss, 5.0, nsteps=50)
    assert abs(sol_v.profiles(out)["ip"] / 1.0e6 - 1.0) < 1e-10
    ss_v = sol_v.steady_state(CurrentState.uniform_faces(50))
    assert abs(ss_v.meta["ip"] / 1.0e6 - 1.0) < 1e-12


@pytest.mark.parametrize("suffix", [".json", ".npz", ".h5"])
def test_state_round_trip_and_restart(tmp_path, suffix):
    sol = _solver()
    s0 = _start(sol)
    s0.meta["note"] = "synthetic"
    s1 = sol.advance(s0, 0.1, nsteps=10)
    path = tmp_path / ("state" + suffix)
    s1.save(path)
    r = CurrentState.load(path)
    np.testing.assert_array_equal(r.psi, s1.psi)
    np.testing.assert_array_equal(r.rho_faces, s1.rho_faces)
    assert r.psi_boundary == s1.psi_boundary and r.time == s1.time and r.meta == s1.meta
    # persistence: continuing from the reloaded state is bit-identical to continuing from the in-memory one
    np.testing.assert_array_equal(sol.advance(s1, 0.2, nsteps=10).psi, sol.advance(r, 0.2, nsteps=10).psi)
    assert CurrentState.from_dict(s1.to_dict()).psi.tolist() == s1.psi.tolist()
    # splitting an interval in two calls equals one call with the same steps up to round-off: the
    # stage times t0 + k h differ in the last bit between the two time grids (0.1 + 0.01 k vs 0.01 (10 + k))
    through = sol.advance(s0, 0.2, nsteps=20)
    restarted = sol.advance(r, 0.2, nsteps=10)
    np.testing.assert_allclose(restarted.psi, through.psi, rtol=0, atol=1e-13 * np.max(np.abs(through.psi)))


def test_sawtooth_hooks_are_called_and_identity_is_neutral():
    calls = []

    def post(state, t, solver):
        calls.append(t)
        return state

    sol = _solver()
    s0 = _start(sol)
    plain = sol.advance(s0, 0.1, nsteps=8)
    hooked = sol.advance(s0, 0.1, nsteps=8, post_step=post)
    assert len(calls) == 8 and abs(calls[-1] - 0.1) < 1e-15
    np.testing.assert_array_equal(plain.psi, hooked.psi)
    ident = _solver(resistivity_modifier=lambda t, rho, eta, state: eta)
    np.testing.assert_allclose(ident.advance(s0, 0.1, nsteps=8).psi, plain.psi, rtol=0, atol=1e-12 * np.ptp(plain.psi))

    # a Jardin-type flattening of eta inside rho < 0.3 must change the core current and leave Ip fixed
    def flatten(t, rho, eta, state):
        out = eta.copy()
        out[rho < 0.3] = np.interp(0.3, rho, eta)
        return out

    mod = _solver(resistivity_modifier=flatten)
    m = mod.advance(s0, 0.1, nsteps=8)
    pm, pp = mod.profiles(m), sol.profiles(plain)
    assert abs(pm["ip"] / pp["ip"] - 1) < 1e-13
    assert np.max(np.abs(pm["q"][:5] - pp["q"][:5])) > 1e-4
