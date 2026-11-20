#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""The opt-in speed settings (``OpenFUSIONToolkit.TokaMaker.kinetic.speed``): validation, the default path never entering them, and the
worker-process mechanics (results, errors and a dead worker) without the solver.

The worker tests fork small processes that do no computation; they need ``os.fork``.

Ported from the reconstruction's own tests (bodies, bounds and references unchanged; imports re-pointed to this package). Not ported here (they need the reconstruction's time-dependent series driver, its EFIT k-file adapter or its forked Jacobian-column workers and their checks, none of which is part of this package, or they import the package without OpenFUSIONToolkit): test_default_is_serial_and_check_returns_none, test_threaded_solver_is_refused, test_worker_sends_results_in_order_and_receives_messages, test_worker_exception_is_reported_not_raised_here, test_dead_worker_gives_none_and_close_reaps_a_running_one, test_effects_round_trip_on_a_stand_in_model, _interval_solver, _run_interval, test_memo_leaves_an_interval_advance_with_the_floor_unchanged.
"""
import os
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from OpenFUSIONToolkit.TokaMaker.kinetic import speed as SP

_FORK = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_invalid_worker_counts_are_refused(bad):
    with pytest.raises(ValueError):
        SP.SpeedSettings(column_workers=bad)


def test_wrong_type_is_refused():
    with pytest.raises(TypeError):
        SP.check_speed(dict(column_workers=2), SimpleNamespace(settings=SimpleNamespace(nthreads=1)))


def test_options_validation_and_threads():
    fm = SimpleNamespace(settings=SimpleNamespace(nthreads=2))
    sp = SP.SpeedSettings(memo=True)
    assert sp.active and not sp.parallel
    assert SP.check_speed(sp, fm) is sp                  # threads matter only for forked workers
    with pytest.raises(ValueError):
        SP.SpeedSettings(memo=1)
    a = SP.SpeedSettings.all_on(4)
    assert (a.column_workers, a.memo, a.mesh_index, a.reuse_stats) == (4, True, True, True)


def test_scope_sets_and_restores_the_process_options():
    from OpenFUSIONToolkit.TokaMaker.kinetic import diagnostics as D
    from OpenFUSIONToolkit.TokaMaker.kinetic import forward as FW
    from OpenFUSIONToolkit.TokaMaker.kinetic import diffusion as DF
    before = (DF._SPEED_MEMO["on"], D.MESH_INDEX["on"], FW.SPEED_OPTIONS["reuse_stats"])
    assert before == (False, False, False)
    with SP.speed_scope(SP.SpeedSettings.all_on(1)):
        assert (DF._SPEED_MEMO["on"], D.MESH_INDEX["on"], FW.SPEED_OPTIONS["reuse_stats"]) == (True, True, True)
        with SP.speed_scope(SP.SpeedSettings(memo=True, mesh_index=False)):
            assert (DF._SPEED_MEMO["on"], D.MESH_INDEX["on"]) == (True, False)
        assert D.MESH_INDEX["on"]
    with pytest.raises(RuntimeError):
        with SP.speed_scope(SP.SpeedSettings(memo=True)):
            raise RuntimeError("leaves the scope")
    assert (DF._SPEED_MEMO["on"], D.MESH_INDEX["on"], FW.SPEED_OPTIONS["reuse_stats"]) == before


@pytest.mark.parametrize("form", ["2", "3"])
def test_profile_memo_returns_the_same_values_and_fresh_arrays(form):
    from OpenFUSIONToolkit.TokaMaker.kinetic import diffusion as DF
    rho = np.linspace(0.0, 1.0, 11)
    vals = np.exp(-3 * rho) + 0.1
    spec = (rho, vals) if form == "2" else ((0.0, 0.02), rho, np.vstack((vals, 2 * vals)))
    f = DF.as_profile_function(spec, log=True)
    pts = np.linspace(0.05, 0.95, 37)
    times = [0.0, 0.007, 0.013, 0.02]
    off = [f(t, pts) for t in times]
    with SP.speed_scope(SP.SpeedSettings(memo=True)):
        on1 = [f(t, pts) for t in times]
        on1[0][:] = -1.0                                 # a caller's change must not reach the stored value
        on2 = [f(t, pts) for t in times]
        with pytest.raises(ValueError):
            f(0.01, np.array([1.5]))                    # the range check still applies
    assert all(np.array_equal(a, b) for a, b in zip(off, on2))
    assert all(np.array_equal(a, b) for a, b in zip(off[1:], on1[1:]))
