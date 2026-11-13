import os
import sys

import numpy as np
import pytest

test_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(test_dir, '..')))
from oft_testing import run_OFT

oft_in = """
&runtime_options
 ppn=1
 debug=0
 test_run=T
/
"""


def clip(poly, vals):
    '''Keep the part of a convex polygon (barycentric vertices) where a linear function (vertex values) is positive'''
    out = []
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        if vals[i] > 0.0:
            out.append(poly[i])
        if (vals[i] > 0.0) != (vals[j] > 0.0):
            s = vals[i]/(vals[i] - vals[j])
            out.append(poly[i] + s*(poly[j] - poly[i]))
    return out


def triangle_rule(n=10):
    '''Collapsed Gauss-Legendre rule on the reference triangle (exact to degree 2n-2), weights summing to one'''
    x, w = np.polynomial.legendre.leggauss(n)
    x = 0.5*(x + 1.0)
    w = 0.5*w
    pts, wts = [], []
    for xi, wi in zip(x, w):
        for eta, we in zip(x, w):
            pts.append([xi, (1.0 - xi)*eta])
            wts.append(2.0*wi*we*(1.0 - xi))
    return np.array(pts), np.array(wts)


def poly_f(l1, l2):
    return 1.0 + 2.0*l1 - l2 + 3.0*l1**2 + l1*l2 - 2.0*l2**3 + (l1*l2)**2 + l1**5


def reference(dv, hv):
    '''Area fraction and integral of poly_f over {dv > 0} and every {hv_k > 0}, by independent clipping'''
    poly = [np.eye(3)[i] for i in range(3)]
    poly = clip(poly, list(dv))
    for h in hv:
        if len(poly) < 3:
            break
        poly = clip(poly, [float(np.dot(p, h)) for p in poly])
    if len(poly) < 3:
        return 0.0, 0.0
    qp, qw = triangle_rule()
    area, integral = 0.0, 0.0
    a = poly[0]
    for k in range(1, len(poly) - 1):
        b, c = poly[k], poly[k + 1]
        frac = (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])
        l1 = a[0] + (b[0] - a[0])*qp[:, 0] + (c[0] - a[0])*qp[:, 1]
        l2 = a[1] + (b[1] - a[1])*qp[:, 0] + (c[1] - a[1])*qp[:, 1]
        area += frac
        integral += frac*np.sum(qw*poly_f(l1, l2))
    return area, integral


@pytest.mark.coverage
def test_cut_cell_rule():
    os.chdir(test_dir)
    with open('oft.in', 'w+') as fid:
        fid.write(oft_in)
    assert run_OFT('./test_gs_cut_cell', 1, 120)
    # A wholly-inside cell gets exactly the cell's rule
    whole = np.loadtxt('gs_cut_cell_whole.results')
    assert np.all(whole[:, 2] == 0.0) and np.all(whole[:, 3] == 0.0)
    data = np.loadtxt('gs_cut_cell.results')
    for order in (5, 9):
        rows = data[data[:, 0] == order]
        for row in rows:
            npts, nplanes = int(row[3]), int(row[4])
            dv, hv = row[9:12], [row[12:15], row[15:18]][:nplanes]
            area, integral = reference(dv, hv)
            # The rule integrates polynomials up to its order exactly over the clipped region
            assert row[6] == pytest.approx(area, abs=1.0E-13)
            assert row[7] == pytest.approx(integral, abs=1.0E-12)
        # Continuity in the swept parameter: refining the sweep tenfold shrinks the largest step change tenfold
        for sweep in range(1, 7):
            steps = []
            for ires in (1, 2):
                sel = rows[(rows[:, 1] == sweep) & (rows[:, 2] == ires)]
                sel = sel[np.argsort(sel[:, 5])]
                steps.append(np.max(np.abs(np.diff(sel[:, 8]))))
            assert steps[0] > 0.0
            assert steps[1] < 0.2*steps[0]
