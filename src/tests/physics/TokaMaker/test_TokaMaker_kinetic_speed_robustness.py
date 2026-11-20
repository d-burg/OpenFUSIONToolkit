#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""Robustness of the opt-in speed settings (``OpenFUSIONToolkit.TokaMaker.kinetic.speed``) without the solver: the time bound of a worker
column, the thread check on the solver's own environment, the refusal off Linux, a worker that cannot be made, the
workers' temporary directories, the mesh caches keyed on the mesh itself, C-stdio text across a fork, and the memo's
cached values (a mutated hit, a time-dependent input on a static geometry, a solver copy with other inputs).

The worker tests fork small processes that do no computation; they need ``os.fork``.

Ported from the reconstruction's own tests (bodies, bounds and references unchanged; imports re-pointed to this package). Not ported here (they need the reconstruction's time-dependent series driver, its EFIT k-file adapter or its forked Jacobian-column workers and their checks, none of which is part of this package, or they import the package without OpenFUSIONToolkit): test_threads_are_read_from_the_solver_environment_not_the_settings, test_parallel_columns_are_refused_off_linux, test_a_silent_worker_times_out_and_close_kills_it, test_a_result_within_the_bound_is_read, test_a_refused_fork_raises_oserror_and_leaves_nothing, test_stale_worker_directories_of_dead_processes_are_removed_and_nothing_else, test_worker_directory_is_named_after_this_process_and_removed, test_c_stdio_text_is_flushed_before_the_fork_and_written_once.
"""
import copy
import os
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import pytest

from OpenFUSIONToolkit.TokaMaker.kinetic import diagnostics as D
from OpenFUSIONToolkit.TokaMaker.kinetic import speed as SP

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_FORK = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


def _fm(nthreads=1, env=None):
    fm = SimpleNamespace(settings=SimpleNamespace(nthreads=nthreads))
    if env is not None:
        fm.env = env
    return fm


# ---------------------------------------------------------------------------------------------------- settings
def test_time_bound_settings_and_rule(monkeypatch):
    s = SP.SpeedSettings()
    assert (s.column_time_factor, s.column_time_floor_s) == (3.0, 60.0)
    assert s.as_dict()["column_time_factor"] == 3.0 and s.as_dict()["column_time_floor_s"] == 60.0
    for bad in (dict(column_time_factor=0.0), dict(column_time_factor=-1.0), dict(column_time_factor=float("nan")),
                dict(column_time_factor=True), dict(column_time_floor_s=-1.0), dict(column_time_floor_s=float("inf")),
                dict(column_time_floor_s="60")):
        with pytest.raises(ValueError):
            SP.SpeedSettings(**bad)
    monkeypatch.setitem(SP._SOLVE_WALL, "longest", 0.0)
    sp = SP.SpeedSettings(column_workers=2, column_time_factor=3.0, column_time_floor_s=60.0)
    assert SP.column_time_bound(sp, 8) == 60.0                     # nothing measured yet: the floor
    for w in (2.0, None, float("nan"), "x", 13.0, 5.0):
        SP.note_solve_wall(w)
    assert SP.longest_solve() == 13.0
    assert SP.column_time_bound(sp, 8) == 3.0 * 8 * 13.0
    assert SP.column_time_bound(SP.SpeedSettings(column_time_factor=0.01, column_time_floor_s=0.0), 8) == \
        pytest.approx(0.01 * 8 * 13.0)


def test_openmp_query_loads_nothing():
    n = SP._openmp_max_threads()
    assert n is None or (isinstance(n, int) and n >= 1)


def _open_fds():
    for d in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(d):
            return len(os.listdir(d))
    return None


# ---------------------------------------------------------------------------------------------------- temp dirs
def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


# ---------------------------------------------------------------------------------------------------- mesh caches
class _Mesh:
    def __init__(self, xy, tri):
        self.r = np.column_stack((xy, np.zeros(len(xy))))
        self.lc = tri
        self.lim_contour = None


def _mirror_meshes():
    """Two different meshes with equal coordinate sums, equal x-weighted sums and equal shapes: a grid symmetric in z
    with a notch cut from its upper half, and its mirror image."""
    xs, zs = np.arange(0, 9), np.arange(-4, 5)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    xy = np.column_stack((X.ravel(), Z.ravel())).astype(float)
    tri = []
    for i in range(xs.size - 1):
        for j in range(zs.size - 1):
            if 3 <= i <= 4 and zs[j] >= 1:
                continue
            a, b, c, d = i * zs.size + j, (i + 1) * zs.size + j, (i + 1) * zs.size + j + 1, i * zs.size + j + 1
            tri += [(a, b, c), (a, c, d)]
    tri = np.array(tri, dtype=np.int64)
    xy2 = xy.copy()
    xy2[:, 1] *= -1.0
    return _Mesh(xy, tri), _Mesh(xy2, tri), xy2, tri


@pytest.mark.parametrize("index", [False, True])
def test_mesh_caches_tell_two_meshes_with_equal_sums_apart(index, monkeypatch):
    m1, m2, xy2, tri = _mirror_meshes()
    pA = np.array([[1.5, 0.5], [6.5, -2.5]])
    pB = np.array([[4.0, 2.0], [4.0, -2.0]])                    # in the notch of m1 / in the notch of m2
    ref = D._on_mesh_compute_scan(xy2, tri, pB)
    assert ref.tolist() == [True, False]
    monkeypatch.setitem(D.MESH_INDEX, "on", index)
    for first in (pA, pB):                                      # the index cache (pA), the point-set cache (pB)
        D._ON_MESH_CACHE.clear()
        D._MESH_INDEX.clear()
        D.on_mesh(m1, first)
        assert D.on_mesh(m2, pB).tolist() == ref.tolist()
    D._ON_MESH_CACHE.clear()
    D._MESH_INDEX.clear()


# ---------------------------------------------------------------------------------------------------- the memo
@pytest.mark.parametrize("form", ["2", "3"])
def test_profile_memo_hit_is_a_fresh_array(form):
    from OpenFUSIONToolkit.TokaMaker.kinetic import diffusion as DF
    rho = np.linspace(0.0, 1.0, 11)
    vals = np.exp(-3 * rho) + 0.1
    spec = (rho, vals) if form == "2" else ((0.0, 0.02), rho, np.vstack((vals, 2 * vals)))
    f = DF.as_profile_function(spec, log=True)
    pts = np.linspace(0.05, 0.95, 37)
    off = f(0.007, pts)
    with SP.speed_scope(SP.SpeedSettings(memo=True)):
        f(0.007, pts)                                           # stored
        hit = f(0.007, pts)                                     # a hit
        hit[:] = -1.0                                           # a caller's change of a hit must not reach the memo
        again = f(0.007, pts)
    assert np.array_equal(again, off)
    assert not np.shares_memory(hit, again)


def _static_solver():
    """A static geometry with a time-dependent conductivity, <J_NI.B> and Ip boundary: the geometry object is the
    same at every time, so only the time tells the coefficients of two times apart."""
    from OpenFUSIONToolkit.TokaMaker.kinetic import diffusion as DF
    from OpenFUSIONToolkit.TokaMaker.kinetic.geometry import FluxSurfaceGeometry
    g = FluxSurfaceGeometry.cylinder(0.6, 1.7, 2.0, n=41)
    rho = g.rho
    sig = np.vstack((3e7 * (1.0 - 0.95 * rho ** 2) ** 3 + 1e5, 3.3e7 * (1.0 - 0.95 * rho ** 2) ** 3 + 1e5))
    jni = np.vstack((2e5 * rho * np.exp(-4 * (rho - 0.7) ** 2), 3.1e5 * rho * np.exp(-4 * (rho - 0.6) ** 2)))
    return DF.CurrentDiffusion(g, ((0.0, 0.02), rho, sig), DF.IpBoundary(((0.0, 0.02), (1.5e6, 1.62e6))),
                               j_ni=((0.0, 0.02), rho, jni)), rho


def _coeff_values(sol, rf, times):
    out = []
    for t in times:
        c = sol.coefficients(t, rf)
        out += [c.jni_c.copy(), c.s_c.copy(), c.m_c.copy(), c.k_f.copy(), np.array([float(c.bc_value)])]
    return out


def test_memo_on_a_static_geometry_keeps_time_dependent_inputs_apart():
    from OpenFUSIONToolkit.TokaMaker.kinetic import diffusion as DF
    times = [0.0, 0.007, 0.013, 0.02, 0.007, 0.0]
    sol, _ = _static_solver()
    rf = DF.CurrentState.uniform_faces(24)
    off = _coeff_values(sol, rf, times)
    st0 = sol.steady_state(rf, t=0.0)
    end_off = sol.advance(st0, 0.02, dt=0.001)
    sol_on, _ = _static_solver()
    with SP.speed_scope(SP.SpeedSettings(memo=True)):
        on = _coeff_values(sol_on, rf, times)
        end_on = sol_on.advance(sol_on.steady_state(rf, t=0.0), 0.02, dt=0.001)
        end_on2 = sol_on.advance(sol_on.steady_state(rf, t=0.0), 0.02, dt=0.001)   # the second pass hits
    assert not np.array_equal(off[0], off[5])                   # the inputs do depend on time
    assert len(off) == len(on) and all(np.array_equal(a, b) for a, b in zip(off, on))
    for end in (end_on, end_on2):
        assert np.array_equal(end_off.psi, end.psi) and end_off.psi_boundary == end.psi_boundary


def test_memo_on_a_solver_copy_with_other_inputs_uses_those_inputs():
    from OpenFUSIONToolkit.TokaMaker.kinetic import diffusion as DF
    sol, rho = _static_solver()
    rf = DF.CurrentState.uniform_faces(24)
    other_jni = DF.as_profile_function(((0.0, 0.02), rho, np.vstack((1e5 * rho, 1.5e5 * rho))))
    other_bnd = DF.IpBoundary(((0.0, 0.02), (1.1e6, 1.2e6)))

    def copies(s):
        a = copy.copy(s)                                        # a copy shares the solver's caches
        a.jni_fn = other_jni
        b = copy.copy(s)
        b.boundary = other_bnd
        return a, b

    ref_a, ref_b = copies(sol)
    ref = (_coeff_values(ref_a, rf, [0.007]), _coeff_values(ref_b, rf, [0.007]))
    sol_on, _ = _static_solver()
    with SP.speed_scope(SP.SpeedSettings(memo=True)):
        base = _coeff_values(sol_on, rf, [0.007])               # stored for the original inputs
        a, b = copies(sol_on)
        got = (_coeff_values(a, rf, [0.007]), _coeff_values(b, rf, [0.007]))
    assert not np.array_equal(base[0], ref[0][0])               # the copy's <J_NI.B> differs from the original's
    assert not np.array_equal(base[4], ref[1][4])               # the copy's Ip boundary value differs too
    for r, g in zip(ref, got):
        assert len(r) == len(g) and all(np.array_equal(x, y) for x, y in zip(r, g))
