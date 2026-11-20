#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
r'''! Speed settings of the reconstruction (:class:`SpeedSettings`); none of them changes a result

Defaults.  With no ``speed`` argument, :func:`.fit.reconstruct` runs with ``SpeedSettings()``: the three
in-process options -- the memo of pure evaluations (``memo``), the indexed point location (``mesh_index``) and
reused statistics (``reuse_stats``) -- on, and one process.  Each in-process option gives the same result bit for
bit as without it.  ``speed=SpeedSettings(...)`` replaces the default whole; ``speed=SpeedSettings.all_off()`` turns
every option off (the behavior before these options became the default, for bitwise comparisons with older runs and
for debugging).  Every fit record holds the options in effect (``speed_in_effect``, :func:`in_effect`).

Parallel Jacobian columns in forked worker processes (``column_workers > 1``) are not part of this package: such a
setting is refused when a fit starts (:func:`check_speed`).  The fields ``column_workers``, ``column_time_factor``
and ``column_time_floor_s`` are kept so that the settings and their record have one form; with
``column_workers = 1`` the two time-bound fields are not used.

For many independent fits, the in-process options in one process per fit cost the fewest core-seconds.

@authors Daniel Burgess
@date November 2026
@ingroup doxy_oft_python
'''

import contextlib
import ctypes
import os
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["SpeedSettings", "check_speed", "column_time_bound", "effective", "in_effect", "solver_threads",
           "speed_scope", "validate"]


@dataclass(frozen=True)
class SpeedSettings:
    """Speed options of :func:`.fit.reconstruct` (module docstring).  None changes a value.  ``SpeedSettings()`` is
    the default (used when ``speed`` is not given): the three in-process options on, one process.  A given
    ``SpeedSettings`` replaces the default whole (each field as given).  :meth:`all_off` turns every option off (the
    behavior before the in-process options became the default: for bitwise comparisons with older runs and for
    debugging).

    ``column_workers``: processes that evaluate the columns of a Jacobian at once; only 1 (serial, the default) is
    accepted by this package (values > 1 are refused when a fit starts, :func:`check_speed`).
    ``memo`` (default on): the memo of pure evaluations in the current model
    (:func:`.diffusion.set_speed_memo`).
    ``mesh_index`` (default on): the bucket-indexed exact point-in-triangle test (:func:`.diagnostics.on_mesh`).
    ``reuse_stats`` (default on): a forward solve reuses the equilibrium statistics of its latest attempt instead of
    evaluating them again on the same state (:meth:`.forward.ForwardModel.solve`).
    ``column_time_factor`` and ``column_time_floor_s``: the time bound of a worker column
    (:func:`column_time_bound`); kept for the settings' one form and record, not used with ``column_workers = 1``."""
    column_workers: int = 1
    memo: bool = True
    mesh_index: bool = True
    reuse_stats: bool = True
    column_time_factor: float = 3.0
    column_time_floor_s: float = 60.0

    def __post_init__(self):
        if isinstance(self.column_workers, bool) or not isinstance(self.column_workers, (int, np.integer)) \
                or int(self.column_workers) < 1:
            raise ValueError("column_workers must be an integer >= 1")
        for name in ("memo", "mesh_index", "reuse_stats"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be True or False")
        for name in ("column_time_factor", "column_time_floor_s"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v) \
                    or v < 0 or (name == "column_time_factor" and v == 0):
                raise ValueError(f"{name} must be a finite number ({'> 0' if name == 'column_time_factor' else '>= 0'})")

    @classmethod
    def all_on(cls, column_workers):
        """Every option on, with ``column_workers`` processes for the columns."""
        return cls(column_workers=column_workers, memo=True, mesh_index=True, reuse_stats=True)

    @classmethod
    def all_off(cls):
        """Every option off: one process, no memo, the plain point location, statistics evaluated on every attempt
        (as before the in-process options became the default; for bitwise comparisons and debugging)."""
        return cls(column_workers=1, memo=False, mesh_index=False, reuse_stats=False)

    @property
    def parallel(self):
        return int(self.column_workers) > 1

    @property
    def active(self):
        return self.parallel or self.memo or self.mesh_index or self.reuse_stats

    def as_dict(self):
        return asdict(self)


def _openmp_max_threads():
    """``omp_get_max_threads()`` of an OpenMP runtime already loaded into this process (the threads the next parallel
    region of the calling thread would use), or None when none is loaded (only an already loaded library is asked;
    nothing is loaded here)."""
    mode = getattr(os, "RTLD_NOLOAD", None)
    if mode is None:
        return None
    for name in ("libgomp.so.1", "libomp.so", "libomp.so.5", "libiomp5.so"):
        try:
            lib = ctypes.CDLL(name, mode=mode | getattr(os, "RTLD_LAZY", 1))
            fn = lib.omp_get_max_threads
        except (OSError, AttributeError):
            continue
        fn.restype = ctypes.c_int
        fn.argtypes = []
        return int(fn())
    return None


def solver_threads(fm):
    """The OpenMP thread count the forward model's solver runs with, and where it was read.

    The forward model's OFT environment (``fm.env``; when one is given to :class:`.forward.ForwardModel`,
    ``ForwardSettings.nthreads`` is not used) holds the count it set at its start, and the loaded OpenMP runtime
    holds the count in effect now (``OFT_env.set_num_threads`` changes it later); the larger of the two is returned.
    Without an environment (a stand-in model) ``ForwardSettings.nthreads`` is the count.  Returns ``(count or None
    when it cannot be established, description)``."""
    env = getattr(fm, "env", None)
    if env is None:
        nt = getattr(getattr(fm, "settings", None), "nthreads", 1)
        return int(nt), f"ForwardSettings.nthreads = {nt} (the forward model has no solver environment)"
    found = {}
    nt_env = getattr(env, "nthreads", None)
    if nt_env is not None and int(nt_env) > 0:            # <= 0: the runtime's default, read from the runtime
        found["the solver environment's nthreads"] = int(nt_env)
    rt = _openmp_max_threads()
    if rt is not None:
        found["the OpenMP runtime's omp_get_max_threads()"] = rt
    if not found:
        return None, (f"the solver environment's nthreads is {nt_env!r} and no OpenMP runtime could be asked")
    return max(found.values()), "; ".join(f"{k} = {v}" for k, v in found.items())


def check_speed(speed, fm):
    """``speed`` when it asks for anything, else None.  Parallel Jacobian columns (``column_workers > 1``) are
    refused: the forked workers are not part of this package."""
    if speed is None:
        return None
    if not isinstance(speed, SpeedSettings):
        raise TypeError("speed must be a SpeedSettings")
    if not speed.active:
        return None
    if speed.parallel:
        raise ValueError("parallel Jacobian columns (column_workers > 1, forked worker processes) are not part of this "
                         "package; use column_workers=1 (the in-process speed options work on every platform)")
    return speed


def effective(speed):
    """The settings an entry point runs with: ``speed`` when given, else the default ``SpeedSettings()`` (the three
    in-process options on, one process)."""
    if speed is None:
        return SpeedSettings()
    if not isinstance(speed, SpeedSettings):
        raise TypeError("speed must be a SpeedSettings")
    return speed


def validate(speed, fm):
    """:func:`check_speed`.  Called by :func:`.fit.reconstruct` when it starts, before anything is solved."""
    return check_speed(speed, fm)


def in_effect(speed):
    """The speed options in effect, as written into every fit record (``speed_in_effect``): ``memo``, ``mesh_index``
    and ``reuse_stats`` read from the process-wide switches at the moment of the call (inside the entry point's
    :func:`speed_scope`, so what the run actually uses), and under ``workers`` the worker settings of ``speed``
    (``column_workers`` and the time bound; ``column_workers = 1``: one process).  A record only."""
    from . import diagnostics as _diag
    from . import forward as _fw
    from . import diffusion as _dif
    sp = speed if isinstance(speed, SpeedSettings) else SpeedSettings()
    return dict(memo=bool(_dif._SPEED_MEMO["on"]), mesh_index=bool(_diag.MESH_INDEX["on"]),
                reuse_stats=bool(_fw.SPEED_OPTIONS["reuse_stats"]),
                workers=dict(column_workers=int(sp.column_workers), column_time_factor=float(sp.column_time_factor),
                             column_time_floor_s=float(sp.column_time_floor_s)))


# ============================================================== the time bound of a worker column
#: The longest consistent solve (wall seconds) made so far in this process.
_SOLVE_WALL = dict(longest=0.0)


def note_solve_wall(seconds):
    """Record the wall time of a consistent solve (only the longest is kept)."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return
    if np.isfinite(s) and s > _SOLVE_WALL["longest"]:
        _SOLVE_WALL["longest"] = s


def longest_solve():
    """The longest consistent solve noted in this process so far [s]."""
    return float(_SOLVE_WALL["longest"])


def column_time_bound(speed, n_col):
    """Wall seconds a worker column may run:
    ``max(column_time_floor_s, column_time_factor * n_col * longest consistent solve so far)``
    (:class:`SpeedSettings`; ``n_col``: the most consistent solves one column can make)."""
    return max(float(speed.column_time_floor_s), float(speed.column_time_factor) * int(n_col) * longest_solve())


@contextlib.contextmanager
def speed_scope(speed):
    """Set the process-wide options of ``speed`` (``memo``, ``mesh_index``, ``reuse_stats``, each on or off as given)
    for the duration of the block and restore the previous values afterwards; ``None`` leaves them as they are.
    :func:`.fit.reconstruct` does this itself with its settings (the default ``SpeedSettings()`` when none is given).
    Outside it the switches are off unless a scope sets them: wrap the forward model's construction too, so that its
    row set-up uses ``mesh_index``."""
    if speed is None:
        yield
        return
    from . import diagnostics as _diag
    from . import forward as _fw
    from . import diffusion as _dif
    prev = (_dif.set_speed_memo(speed.memo), _diag.MESH_INDEX["on"], _fw.SPEED_OPTIONS["reuse_stats"])
    _diag.MESH_INDEX["on"] = bool(speed.mesh_index)
    _fw.SPEED_OPTIONS["reuse_stats"] = bool(speed.reuse_stats)
    try:
        yield
    finally:
        _dif.set_speed_memo(prev[0])
        _diag.MESH_INDEX["on"] = prev[1]
        _fw.SPEED_OPTIONS["reuse_stats"] = prev[2]
