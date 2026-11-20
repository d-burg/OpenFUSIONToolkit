#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
"""Second robustness pass of the speed settings, without the solver: the mesh key computed once per mesh and never
stale, a stale-directory cleanup that cannot raise, worker pipes numbered 1024 and above, a time bound over a whole
message, the speed settings in a series record without workers, and the boot in the host tag.

The worker tests fork small processes that do no computation; they need ``os.fork``.

Ported from the reconstruction's own tests (bodies, bounds and references unchanged; imports re-pointed to this package). Not ported here (they need the reconstruction's time-dependent series driver, its EFIT k-file adapter or its forked Jacobian-column workers and their checks, none of which is part of this package, or they import the package without OpenFUSIONToolkit): test_stale_cleanup_cannot_raise_on_planted_names, test_a_worker_is_made_beside_a_planted_name, test_worker_pipes_numbered_1024_and_above, test_time_bound_covers_a_message_that_stopped_half_way, test_a_whole_message_sent_in_pieces_within_the_bound_is_read, test_series_records_the_settings_only_when_given_without_workers, test_host_tag_includes_the_boot_and_falls_back_to_the_host_name, test_boot_id_is_read_or_empty.
"""
import os
import resource
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from OpenFUSIONToolkit.TokaMaker.kinetic import diagnostics as D
from OpenFUSIONToolkit.TokaMaker.kinetic import speed as SP

_FORK = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")


class _Mesh:
    def __init__(self, xy, tri):
        self.r = np.column_stack((xy, np.zeros(len(xy))))
        self.lc = tri
        self.lim_contour = None


def _notched_grid():
    """A grid symmetric in z with a notch cut from its upper half (its mirror image has equal coordinate sums)."""
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
    return xy, np.array(tri, dtype=np.int64)


_PB = np.array([[4.0, 2.0], [4.0, -2.0]])                       # in the upper notch / in the lower half


def _fresh(monkeypatch, index):
    monkeypatch.setitem(D.MESH_INDEX, "on", index)
    monkeypatch.setattr(D, "_ON_MESH_CACHE", {})
    monkeypatch.setattr(D, "_MESH_INDEX", {})
    monkeypatch.setattr(D, "_MESH_KEYS", {})


# ---------------------------------------------------------------------------------------------------- 1. mesh key
def test_mesh_key_is_computed_once_per_mesh(monkeypatch):
    _fresh(monkeypatch, False)
    calls = []
    real = D._mesh_key
    monkeypatch.setattr(D, "_mesh_key", lambda xy, tri: (calls.append(1), real(xy, tri))[1])
    xy, tri = _notched_grid()
    m = _Mesh(xy, tri)
    rng = np.random.default_rng(3)
    for _ in range(6):
        pts = rng.random((5, 2)) * [8.0, 8.0] - [0.0, 4.0]
        assert np.array_equal(D.on_mesh(m, pts), D._on_mesh_compute_scan(xy, tri, pts))
    assert len(calls) == 1


@pytest.mark.parametrize("index", [False, True])
def test_mesh_key_follows_a_mesh_changed_in_place_or_replaced(index, monkeypatch):
    _fresh(monkeypatch, index)
    xy, tri = _notched_grid()
    m = _Mesh(xy, tri)
    assert D.on_mesh(m, _PB).tolist() == [False, True]
    m.r[:, 1] *= -1.0                                           # the same arrays, changed in place: the mirror mesh
    assert D.on_mesh(m, _PB).tolist() == [True, False]
    m.lc[:] = m.lc[::-1].copy()                                 # triangles reordered in place: same answer, new key
    assert D.on_mesh(m, _PB).tolist() == [True, False]
    m.r = np.column_stack((xy, np.zeros(len(xy))))              # replaced by a new array: the original mesh
    assert D.on_mesh(m, _PB).tolist() == [False, True]
    keys = {k[2] for k in D._ON_MESH_CACHE}
    assert len(keys) == 2                                       # the original and the mirror points
    del m
    m2 = _Mesh(*_notched_grid())                                # new objects (perhaps at a freed id): no stale key
    m2.r[:, 1] *= -1.0
    assert D.on_mesh(m2, _PB).tolist() == [True, False]


def test_mesh_key_of_equals_the_digest_key():
    xy, tri = _notched_grid()
    m = _Mesh(xy, tri)
    assert D._mesh_key_of(m.r, m.lc) == D._mesh_key(m.r[:, :2], m.lc)
    assert D._mesh_key_of(m.r, m.lc) == D._mesh_key(np.ascontiguousarray(xy), tri)
