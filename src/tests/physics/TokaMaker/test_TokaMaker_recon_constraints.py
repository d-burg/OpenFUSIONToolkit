#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
# TokaMaker tests: reconstruction constraint types (MSE, finite-length Mirnov, coil currents, used flag,
# off-mesh report, constraint file precision), checked on the solved ITER reference equilibrium and in pure Python
import contextlib
import ctypes
import io
from unittest.mock import Mock

import numpy as np
import pytest

from tokamaker_testing import mp_run, solve_ITER_reference
from OpenFUSIONToolkit.TokaMaker.reconstruction import MSE_con, MSE_con_id, reconstruction

# Measurement points inside the plasma of the ITER reference equilibrium (exact in the constraint file format)
mse_points = np.array([[5.0, 0.5], [6.0, 0.5], [7.0, 0.5], [7.75, 0.25], [6.5, 1.5]])
# Two coefficient sets: a two-term set (c_0 B_Z / c_6 B_phi) and one with all nine terms nonzero
mse_coef_sets = [
    [0.75, 0.0, 0.0, 0.0, 0.0, 0.0, -0.625, 0.0, 0.0],
    [0.8, -0.1, 0.05, 2.5E-7, 0.15, 0.2, -0.9, -1.5E-7, 3.0E-7]
]
# Electric field (E_R, E_Z) [V/m]: off, and on with a size that changes tan(gamma) by several percent
mse_efields = [(0.0, 0.0), (4.0E4, -1.5E4)]


def mse_formula(B, coefs, E_R, E_Z):
    '''tan(gamma) from (B_R, B_phi, B_Z) in the IMAS coefficient order, evaluated as the Fortran does'''
    c = coefs
    num = c[0]*B[2] + c[1]*B[0] + c[2]*B[1] + c[3]*E_R
    den = c[4]*B[2] + c[5]*B[0] + c[6]*B[1] + c[7]*E_R + c[8]*E_Z
    return num/den


def mse_cases():
    '''All (point, coefficient set, electric field, quantity) combinations'''
    cases = []
    for ipt in range(mse_points.shape[0]):
        for icoef in range(len(mse_coef_sets)):
            for ie in range(len(mse_efields)):
                for quantity in ('tan', 'angle'):
                    cases.append((ipt, icoef, ie, quantity))
    return cases


#============================================================================
# Solver checks on the ITER reference equilibrium
#============================================================================
def run_mse_case(mp_q):
    mygs, _ = solve_ITER_reference(1.0, 2)
    if mygs is None:
        mp_q.put(None)
        return
    myrecon = reconstruction(mygs)
    # Field components through the same constraint framework: one Mirnov row per (R, phi, Z) unit normal
    for pt in mse_points:
        for norm in np.eye(3):
            myrecon.add_Mirnov(pt, norm, 0.0, 1.0)
    for ipt, icoef, ie, quantity in mse_cases():
        val = 0.1 if quantity == 'tan' else -0.2
        myrecon.add_MSE(mse_points[ipt], mse_coef_sets[icoef], val, 0.01, quantity=quantity,
                        E_R=mse_efields[ie][0], E_Z=mse_efields[ie][1])
    # Last row: a denominator that is exactly zero (only electric field terms, which cancel), written to the file
    # without the Python check
    zero_den = MSE_con()
    zero_den.loc, zero_den.coefs = (6.0, 0.5), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
    zero_den.E_R, zero_den.E_Z, zero_den.val, zero_den.err = 1.0E4, -1.0E4, 0.1, 0.01
    myrecon._mse_cons.append(zero_den)
    myrecon.setup_constraints()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        err_mat = myrecon.eval_error()
    B_eval = mygs.get_field_eval('B').eval(mse_points)
    mp_q.put([{'err_mat': err_mat, 'B_eval': B_eval, 'output': output.getvalue()}])


def test_recon_mse_closed_form():
    '''MSE constraint against the closed form, both quantities, electric field off and on, on a solved equilibrium;
    a zero denominator gives NaN and a warning.

    The reference field is the solver's own: (a) the field components returned by Mirnov constraints at the same
    points (same evaluation path), and (b) `get_field_eval('B')`, which divides by R + 1E-12 and maps the point
    through the cell, so it differs at about 1E-13 relative.
    Measured (release builds, 2 threads, 40 cases): (a) 0 with GCC on x86 (same math library as numpy, no fused
    multiply-add) and up to 1.9E-15 with the Intel compilers; (b) 1.1E-13.
    Bounds: (a) 1E-13: the numerator and denominator are sums of up to five terms with cancellation (condition number
    up to 41 in these cases), followed by a division and an arctangent from the platform's math library, so the
    rounding bound is about 2E-14; (b) 1E-12.
    '''
    results = mp_run(run_mse_case, ())
    assert results is not None
    err_mat = results[0]['err_mat']
    B_eval = results[0]['B_eval']
    npts = mse_points.shape[0]
    B_fit = err_mat[:3*npts, 1].reshape((npts, 3))
    cases = mse_cases()
    assert err_mat.shape[0] == 3*npts + len(cases) + 1
    assert np.isnan(err_mat[-1, 0]) and np.isnan(err_mat[-1, 1])
    assert np.all(np.isfinite(err_mat[:-1, :]))
    assert 'Modeled signal is not finite for constraint row(s) [{0:d}]'.format(err_mat.shape[0]-1) in results[0]['output']
    for i, (ipt, icoef, ie, quantity) in enumerate(cases):
        row = err_mat[3*npts+i, :]
        E_R, E_Z = mse_efields[ie]
        tan_fit = mse_formula(B_fit[ipt, :], mse_coef_sets[icoef], E_R, E_Z)
        tan_eval = mse_formula(B_eval[ipt, :], mse_coef_sets[icoef], E_R, E_Z)
        if quantity == 'tan':
            ref_fit, ref_eval, val = tan_fit, tan_eval, 0.1
        else:
            ref_fit, ref_eval, val = np.arctan(tan_fit), np.arctan(tan_eval), -0.2
        assert abs(row[1] - ref_fit) <= 1.E-13*abs(ref_fit)
        assert abs(row[1] - ref_eval) <= 1.E-12*abs(ref_eval)
        # Weighted error (value - model)/err, the stored value, and no 3D correction
        assert row[0] == (val - row[1])*100.0
        assert row[2] == val
        assert row[3] == 0.0
    # The electric field terms matter at the level the test resolves
    for ipt in range(npts):
        tan_off = mse_formula(B_fit[ipt, :], mse_coef_sets[1], 0.0, 0.0)
        tan_on = mse_formula(B_fit[ipt, :], mse_coef_sets[1], *mse_efields[1])
        assert abs(tan_on - tan_off) > 1.E-3*abs(tan_off)


mse_flux_pts = [[8.5, 0.5], [6.25, 4.25], [4.0, 0.5], [6.25, -3.5]]
mse_shift = 2  # MSE chord whose datum is displaced


def run_mse_reconstruct_case(mp_q):
    mygs, _ = solve_ITER_reference(1.0, 2)
    if mygs is None:
        mp_q.put(None)
        return
    myrecon = reconstruction(mygs)
    Ip = mygs.get_stats()['Ip']
    psi_eval = mygs.get_field_eval('PSI')
    flux_vals = [2.0*np.pi*psi_eval.eval(np.array(pt))[0] for pt in mse_flux_pts]

    def set_data(mse_vals, rows, unused=()):
        '''Plasma current, flux loops and the MSE chords in `rows`, those in `unused` not used (row order: flux loops,
        Ip, MSE)'''
        myrecon.reset_constraints()
        myrecon.set_Ip(Ip, 1.E5)
        for pt, val in zip(mse_flux_pts, flux_vals):
            myrecon.add_flux_loop(pt, val, 1.E-2)
        for i in rows:
            myrecon.add_MSE(mse_points[i], mse_coef_sets[1], mse_vals[i], 1.E-3, use=(i not in unused))

    # MSE data: the reference equilibrium's signals, one chord displaced by 20 errors
    all_rows = list(range(mse_points.shape[0]))
    set_data(np.full(len(all_rows), 0.1), all_rows)
    myrecon.setup_constraints()
    mse_data = myrecon.eval_error()[len(mse_flux_pts)+1:, 1].copy()
    mse_data[mse_shift] += 2.E-2
    # Fortran driver fitting the magnetic axis radius; every fit starts from the same equilibrium
    mygs.set_isoflux(None)
    mygs.set_saddles(None)
    mygs.set_targets(R0=mygs.o_point[0], V0=mygs.o_point[1])
    myrecon.settings.fit_Pscale = False
    myrecon.settings.fitR0 = True
    eq_start = mygs.copy_eq()
    fits = {}
    for case, rows, unused in (('without', [i for i in all_rows if i != mse_shift], ()),
                               ('unused', all_rows, (mse_shift,)), ('with', all_rows, ())):
        mygs.replace_eq(eq_start)
        set_data(mse_data, rows, unused)
        flag = myrecon.reconstruct(maxits=20)
        fit_out = np.loadtxt(myrecon.out_file)
        # All chords evaluated on the fitted equilibrium
        set_data(mse_data, all_rows)
        myrecon.setup_constraints()
        fits[case] = {'flag': flag, 'fit_out': fit_out, 'err_mat': myrecon.eval_error()}
    mp_q.put([{'fits': fits, 'mse_data': mse_data}])


def test_recon_mse_in_reconstruct():
    '''MSE constraints drive the Fortran reconstruction: with one chord's datum displaced by 20 errors from the
    reference equilibrium's signal, the fit that uses the chord moves its modeled signal toward the datum, relative to
    the fit without the chord, and the fit with the chord unused equals the fit without it; the output table holds the
    MSE rows as `eval_error` evaluates them.
    Measured (release build, 2 threads, constraint file of 7 or 17 significant digits): distance of the modeled
    tan(gamma) from the displaced datum 2.00000E-2 without the chord and 1.99950E-2 with it (a move of 5.0E-6, far
    above round-off); unused against without 0 (bitwise equal); output table against `eval_error` up to 2.7E-13
    relative (the table holds 13 significant digits).
    Bounds: unused against without 1E-15 relative (a few ulp, as the measured difference is zero); table 3E-12
    relative.
    '''
    results = mp_run(run_mse_reconstruct_case, ())
    assert results is not None
    fits, mse_data = results[0]['fits'], results[0]['mse_data']
    row = len(mse_flux_pts) + 1 + mse_shift
    dist = {case: abs(fit['err_mat'][row, 1] - mse_data[mse_shift]) for case, fit in fits.items()}
    assert dist['with'] < dist['without']
    model_without, model_unused = fits['without']['err_mat'][:, 1], fits['unused']['err_mat'][:, 1]
    assert np.all(np.abs(model_unused - model_without) <= 1.E-15*np.abs(model_without))
    fit_out, err_mat = fits['with']['fit_out'], fits['with']['err_mat']
    assert fit_out.shape == (err_mat.shape[0], 5)
    assert np.all(fit_out[:, 0] == np.arange(1, err_mat.shape[0]+1))
    assert abs(fit_out[row, 2] - err_mat[row, 1]) <= 3.E-12*abs(err_mat[row, 1])


def probe_points(loc, norm, length, n_avg):
    '''Averaging points of a finite-length probe, computed as the Fortran does (see `Mirnov_con`)'''
    tnorm = np.sqrt(norm[0]**2 + norm[2]**2)
    if length >= 0.0:
        t = (norm[0]/tnorm, norm[2]/tnorm)
    else:
        t = (-norm[2]/tnorm, norm[0]/tnorm)
    pts = []
    for k in range(n_avg):
        s = (float(k) - 0.5*float(n_avg-1))*abs(length)/float(n_avg)
        pts.append([loc[0] + s*t[0], loc[1] + s*t[1]])
    return np.array(pts)


# Finite-length probes (center, normal, length, n_avg). The first two have averaging points that are exact binary
# fractions, so point probes at the same points see exactly the same locations.
finite_probes = [
    ([6.0, 2.0], [1.0, 0.0, 0.0], 0.5, 4),
    ([6.0, 2.0], [0.0, 0.0, 1.0], -1.0, 8),
    ([7.0, -1.0], [0.6, 0.0, 0.8], 0.25, 25),
    ([5.0, 0.0], [0.6, 0.0, 0.8], 0.1, 1),
    ([6.0, 2.0], [1.0, 0.0, 0.0], 0.0, 3),
]


def run_finite_probe_case(mp_q):
    mygs, _ = solve_ITER_reference(1.0, 2)
    if mygs is None:
        mp_q.put(None)
        return
    myrecon = reconstruction(mygs)
    for loc, norm, length, n_avg in finite_probes:
        myrecon.add_Mirnov(loc, norm, 0.0, 1.0, length=length, n_avg=n_avg)
    # Point probes at the center of each probe, then at the averaging points of the first two
    for loc, norm, _, _ in finite_probes:
        myrecon.add_Mirnov(loc, norm, 0.0, 1.0)
    for loc, norm, length, n_avg in finite_probes[:2]:
        for pt in probe_points(loc, norm, length, n_avg):
            myrecon.add_Mirnov(pt, norm, 0.0, 1.0)
    myrecon.setup_constraints()
    err_mat = myrecon.eval_error()
    field_eval = mygs.get_field_eval('B')
    B_means = []
    for loc, norm, length, n_avg in finite_probes:
        B_pts = field_eval.eval(probe_points(loc, norm, length, n_avg))
        B_means.append(np.mean(np.dot(B_pts, norm)))
    mp_q.put([{'err_mat': err_mat, 'B_means': np.array(B_means)}])


def test_recon_finite_probe():
    '''Finite-length Mirnov sensors against the mean of the solver's own field over the same points.

    (a) Against point probes at the averaging points (same evaluation path; exact binary points), (b) against
    `get_field_eval('B')` at the points (about 1E-13 relative, see `test_recon_mse_closed_form`), (c) one averaging
    point is the point probe exactly, (d) zero length gives the point probe to round-off, and (e) the averaging
    changes the signal by much more than the bounds.
    Measured (release build, 2 threads): (a) 0 (bitwise equal); (b) 2.0E-13; (c) and (d) 0; (e) 1.2E-4 to 2.1E-2.
    Bounds: (a) 1E-15 relative (a few ulp, as the measured difference is zero); (b) 2E-12 relative.
    '''
    results = mp_run(run_finite_probe_case, ())
    assert results is not None
    err_mat = results[0]['err_mat']
    B_means = results[0]['B_means']
    nprobe = len(finite_probes)
    finite = err_mat[:nprobe, 1]
    center = err_mat[nprobe:2*nprobe, 1]
    i = 2*nprobe
    for j, (loc, norm, length, n_avg) in enumerate(finite_probes[:2]):
        point_mean = 0.0
        for val in err_mat[i:i+n_avg, 1]:  # summed in the Fortran order
            point_mean += val
        point_mean /= n_avg
        assert abs(finite[j] - point_mean) <= 1.E-15*abs(point_mean)
        assert abs(finite[j] - center[j]) > 1.E-6*abs(center[j])
        i += n_avg
    assert i == err_mat.shape[0]
    for j in range(4):
        assert abs(finite[j] - B_means[j]) <= 2.E-12*abs(B_means[j])
    assert finite[3] == center[3]
    assert abs(finite[4] - center[4]) <= 2.3E-16*abs(center[4])


far_point = [40.0, 0.0]  # Outside the ITER mesh


def run_offmesh_case(mp_q):
    mygs, _ = solve_ITER_reference(1.0, 2)
    if mygs is None:
        mp_q.put(None)
        return
    myrecon = reconstruction(mygs)
    myrecon.add_flux_loop([6.0, 0.5], 1.5, 0.01)
    myrecon.add_flux_loop([6.0, 0.5], 1.5, 0.01, use=False)
    myrecon.add_flux_loop(far_point, 1.5, 0.01)
    myrecon.add_Mirnov([6.0, 0.5], [1.0, 0.0, 0.0], 0.25, 0.01)
    myrecon.add_Mirnov([6.0, 0.5], [1.0, 0.0, 0.0], 0.25, 0.01, use=False)
    myrecon.add_Mirnov(far_point, [1.0, 0.0, 0.0], 0.25, 0.01)
    myrecon.add_Mirnov(far_point, [1.0, 0.0, 0.0], 0.25, 0.01, length=0.5, n_avg=4)
    myrecon.add_pressure(far_point, 1.0E3, 10.0)
    myrecon.set_Ip(1.5E7, 1.5E5, use=False)
    myrecon.add_MSE(far_point, mse_coef_sets[0], 0.1, 0.01)
    # Locating before the constraints are set up is an error
    from OpenFUSIONToolkit.TokaMaker.reconstruction import c_double_ptr, tokamaker_recon_locate
    loc_mat = np.zeros((1, 2))
    locate_flag = ctypes.c_int()
    tokamaker_recon_locate(mygs._tMaker_ptr, loc_mat.ctypes.data_as(c_double_ptr), ctypes.byref(locate_flag))
    myrecon.setup_constraints()
    err_mat = myrecon.eval_error()
    projected = [(entry['index'], entry['points'], entry['distance']) for entry in myrecon.projected]
    myrecon.read_fit_in()
    use_read = [con.use for con in myrecon.get_constraints()]
    # Nearest mesh boundary vertex to the far point (boundary edges belong to one cell only)
    from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
    mesh_pts, mesh_lc, _, _, _ = load_gs_mesh('ITER_mesh.h5')
    edges = {}
    for cell in mesh_lc:
        for i in range(3):
            edge = tuple(sorted((cell[i], cell[(i+1) % 3])))
            edges[edge] = edges.get(edge, 0) + 1
    bnodes = np.unique([node for edge, count in edges.items() if count == 1 for node in edge])
    dist = np.sqrt(np.sum((mesh_pts[bnodes, :2] - np.array(far_point))**2, axis=1))
    node = mesh_pts[bnodes[np.argmin(dist)], :2]
    psi_node = mygs.get_field_eval('PSI').eval(node)[0]
    dist_finite = 0.0
    for pt in probe_points(far_point, [1.0, 0.0, 0.0], 0.5, 4):
        dist_finite = max(dist_finite, np.sqrt(np.sum((mesh_pts[bnodes, :2] - pt)**2, axis=1)).min())
    mp_q.put([{'err_mat': err_mat, 'projected': projected, 'use_read': use_read, 'dist': dist.min(),
               'dist_finite': dist_finite, 'psi_node': psi_node, 'locate_flag': locate_flag.value}])


def test_recon_offmesh_and_use():
    '''Off-mesh constraints are reported after setup (row, number of moved points, distance to the nearest boundary
    node), still evaluated at that node; unused constraints have zero weighted error and the same modeled signal as
    their used twins, and the flag survives the constraint file; locating before setup returns an error'''
    results = mp_run(run_offmesh_case, ())
    assert results is not None
    assert results[0]['locate_flag'] == -102
    err_mat = results[0]['err_mat']
    dist = results[0]['dist']
    projected = results[0]['projected']
    assert [(entry[0], entry[1]) for entry in projected] == [(2, 1), (5, 1), (6, 4), (7, 1), (9, 1)]
    for entry in projected:
        dist_exp = results[0]['dist_finite'] if entry[0] == 6 else dist
        assert abs(entry[2] - dist_exp) <= 1.E-14*dist_exp
    assert results[0]['dist_finite'] > dist
    # The moved flux loop is evaluated at the boundary node
    assert abs(err_mat[2, 1] - 2.0*np.pi*results[0]['psi_node']) <= 1.E-12*abs(err_mat[2, 1])
    # Unused rows: zero weighted error, same model as the used twin, stored value kept
    for used, unused in ((0, 1), (3, 4)):
        assert err_mat[unused, 0] == 0.0
        assert err_mat[unused, 1] == err_mat[used, 1]
        assert err_mat[unused, 2] == err_mat[used, 2]
        assert err_mat[used, 0] != 0.0
    assert err_mat[8, 0] == 0.0
    assert results[0]['use_read'] == [True, False, True, True, False, True, True, True, False, True]


#============================================================================
# Pure Python checks (mocked solver)
#============================================================================
@pytest.fixture
def recon(tmp_path):
    '''Reconstruction object with a mocked TokaMaker object, writing its files to `tmp_path`'''
    maker = Mock()
    maker._oft_env.oft_in_groups = {}
    maker.coil_sets = {'PF1': {'id': 0}, 'PF2': {'id': 1}, 'VS': {'id': 2}}
    return reconstruction(maker, in_filename=str(tmp_path / 'fit.in'),
                          out_filename=str(tmp_path / 'fit.out'))


@pytest.mark.parametrize('quantity', ['tan', 'angle'])
def test_recon_mse_round_trip(recon, quantity):
    '''Every MSE field survives writing and reading the constraint file, after the other constraint types'''
    recon.add_flux_loop([6.0, 0.5], 1.5, 0.01)
    recon.set_Ip(1.0E7, 1.0E5)
    recon.add_MSE([6.25, 0.5], mse_coef_sets[1], 0.125, 0.0025, quantity=quantity, E_R=4.0E4, E_Z=-1.5E4, phi=0.5)
    recon.write_fit_in()
    recon.read_fit_in()
    assert len(recon._mse_cons) == 1
    assert len(recon._flux_loops) == 1
    restored = recon._mse_cons[0]
    assert restored.loc == (6.25, 0.5)
    assert restored.phi == 0.5
    assert restored.coefs == tuple(mse_coef_sets[1])
    assert (restored.E_R, restored.E_Z) == (4.0E4, -1.5E4)
    assert restored.quantity == quantity
    assert restored.val == 0.125
    assert restored.err == pytest.approx(0.0025, rel=1.E-15)
    # MSE rows come last in the file and in the error matrix
    assert recon.get_constraints()[-1] is restored


def test_recon_mse_file_record(recon):
    '''The MSE record layout read by the Fortran loader: type, location and phi, nine coefficients, (E_R, E_Z),
    quantity code, value and weight'''
    recon.add_MSE([6.0, 0.5], mse_coef_sets[0], -0.25, 0.5, quantity='angle')
    recon.write_fit_in()
    with open(recon.con_file) as stream:
        lines = stream.read().split('\n')
    assert int(lines[0]) == 1
    assert int(lines[2]) == MSE_con_id
    assert [float(x) for x in lines[3].split()] == [6.0, 0.5, 0.0]
    assert [float(x) for x in lines[4].split()] == mse_coef_sets[0]
    assert [float(x) for x in lines[5].split()] == [0.0, 0.0]
    assert int(lines[6]) == 2
    assert [float(x) for x in lines[7].split()] == [-0.25, 2.0]


@pytest.mark.parametrize('kwargs,match', [
    ({'loc': [6.0]}, 'Location must have shape'),
    ({'loc': [np.nan, 0.5]}, 'Location must be finite'),
    ({'loc': [-6.0, 0.5]}, 'R > 0'),
    ({'loc': None}, 'Location is required'),
    ({'coefs': [1.0]*8}, 'coefficients must have shape'),
    ({'coefs': [1.0]*8 + [np.inf]}, 'coefficients must be finite'),
    ({'coefs': [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, 'denominator is identically zero'),
    ({'coefs': [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]}, 'denominator is identically zero'),
    ({'coefs': [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], 'E_R': 2.0E4, 'E_Z': -2.0E4},
     'denominator is identically zero'),
    ({'coefs': ['a']*9}, 'coefficients must be numeric'),
    ({'coefs': [True]*9}, 'coefficients must be numeric'),
    ({'val': True}, 'MSE value must be numeric'),
    ({'err': True}, 'Error must be numeric'),
    ({'val': np.nan}, 'MSE value must be finite'),
    ({'val': [0.1, 0.2]}, r'MSE value must have shape \(\)'),
    ({'val': 2.0, 'quantity': 'angle'}, r'angle must lie in'),
    ({'err': 0.0}, 'Error must be positive'),
    ({'err': -0.01}, 'Error must be positive'),
    ({'err': np.inf}, 'Error must be finite'),
    ({'quantity': 'gamma'}, 'quantity must be one of'),
    ({'E_R': np.nan}, 'E_R must be finite'),
    ({'E_Z': [1.0, 2.0]}, r'E_Z must have shape \(\)'),
    ({'phi': np.inf}, 'phi must be finite'),
])
def test_recon_mse_rejects_invalid_input(recon, kwargs, match):
    '''Invalid MSE input raises ValueError before a constraint is added'''
    args = {'loc': [6.0, 0.5], 'coefs': mse_coef_sets[1], 'val': 0.1, 'err': 0.01}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        recon.add_MSE(**args)
    assert recon._mse_cons == []


def test_recon_mse_electric_field_denominator(recon):
    '''A denominator made of electric field terms only is accepted when those terms do not cancel'''
    recon.add_MSE([6.0, 0.5], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 0.1, 0.01, E_R=2.0E4)
    recon.add_MSE([6.0, 0.5], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], 0.1, 0.01, E_R=2.0E4, E_Z=-1.0E4)
    assert len(recon._mse_cons) == 2


def test_recon_mse_rejects_invalid_quantity_in_file(recon):
    '''A constraint file with an unknown MSE quantity code is rejected'''
    with open(recon.con_file, 'w') as stream:
        stream.write('1\n\n13\n 6.0 0.5 0.0\n' + ' 1.0'*9 + '\n 0.0 0.0\n 3\n 0.1 100.0\n\n')
    with pytest.raises(ValueError, match='Invalid MSE quantity'):
        recon.read_fit_in()


def test_recon_finite_probe_round_trip(recon):
    '''A finite-length Mirnov sensor is written as a type 14 record and read back; a point probe keeps the type 1
    record of three lines'''
    from OpenFUSIONToolkit.TokaMaker.reconstruction import Mirnov_con_id, finite_Mirnov_con_id
    recon.add_Mirnov([6.0, 2.0], [0.6, 0.0, 0.8], 0.25, 0.01)
    recon.add_Mirnov([6.0, 2.0], [0.6, 0.0, 0.8], 0.25, 0.01, length=-0.125, n_avg=25)
    recon.write_fit_in()
    with open(recon.con_file) as stream:
        lines = stream.read().split('\n')
    assert int(lines[2]) == Mirnov_con_id
    assert lines[6] == ''
    assert int(lines[7]) == finite_Mirnov_con_id
    assert [float(x) for x in lines[10].split()] == [-0.125, 25]
    recon.read_fit_in()
    point, finite = recon._mirnovs
    assert (point.length, point.n_avg) == (None, None)
    assert (finite.length, finite.n_avg) == (-0.125, 25)
    assert finite.loc == (6.0, 2.0)
    assert finite.norm == (0.6, 0.0, 0.8)
    assert (finite.val, finite.err) == (0.25, 0.01)


@pytest.mark.parametrize('kwargs,match', [
    ({'length': 0.1}, 'both `length` and `n_avg`'),
    ({'n_avg': 5}, 'both `length` and `n_avg`'),
    ({'length': np.nan, 'n_avg': 5}, 'Mirnov length must be finite'),
    ({'length': [0.1, 0.2], 'n_avg': 5}, r'Mirnov length must have shape \(\)'),
    ({'length': 0.1, 'n_avg': 0}, 'integer >= 1'),
    ({'length': 0.1, 'n_avg': 2.5}, 'integer >= 1'),
    ({'length': 0.1, 'n_avg': True}, 'integer >= 1'),
    ({'length': True, 'n_avg': 5}, 'Mirnov length must be numeric'),
    ({'length': 0.1, 'n_avg': 5, 'norm': None}, 'need a normal'),
    ({'length': 0.1, 'n_avg': 5, 'norm': [0.0, 1.0, 0.0]}, 'nonzero poloidal'),
    ({'length': 0.1, 'n_avg': 5, 'norm': [np.nan, 0.0, 1.0]}, 'Mirnov normal must be finite'),
])
def test_recon_finite_probe_rejects_invalid_input(recon, kwargs, match):
    '''Invalid finite-length probe input raises ValueError before a constraint is added'''
    args = {'loc': [6.0, 2.0], 'norm': [1.0, 0.0, 0.0], 'val': 0.25, 'err': 0.01}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        recon.add_Mirnov(**args)
    assert recon._mirnovs == []


def test_recon_finite_probe_rejects_invalid_file(recon):
    '''A finite-length probe record with fewer than one averaging point is rejected'''
    with open(recon.con_file, 'w') as stream:
        stream.write('1\n\n14\n 6.0 2.0 0.0\n 1.0 0.0 0.0\n 0.1 0\n 0.25 100.0\n\n')
    with pytest.raises(ValueError, match='n_avg'):
        recon.read_fit_in()


def add_every_type(recon, use):
    '''Add one constraint of every type that can be added from Python'''
    recon.add_flux_loop([6.0, 0.5], 1.5, 0.01, use=use)
    recon.add_Mirnov([6.0, 0.5], [1.0, 0.0, 0.0], 0.25, 0.02, use=use)
    recon.add_Mirnov([6.0, 0.5], [1.0, 0.0, 0.0], 0.25, 0.02, length=0.5, n_avg=4, use=use)
    recon.add_pressure([6.0, 0.5], 1.0E3, 10.0, use=use)
    recon.set_coil_currents({'PF1': 1.0E3, 'PF2': -2.0E3, 'VS': 0.0}, {'PF1': 10.0, 'PF2': 20.0, 'VS': 5.0},
                            use={'PF1': use, 'PF2': use, 'VS': use})
    recon.set_Ip(1.0E7, 1.0E5, use=use)
    recon.set_DFlux(0.05, 0.005, use=use)
    recon.add_MSE([6.0, 0.5], mse_coef_sets[0], 0.1, 0.01, use=use)


def test_recon_use_round_trip(recon):
    '''The used flag and the error survive the constraint file for every constraint type; an unused constraint is
    written with weight -1/err, which the solver reads as weight 0'''
    add_every_type(recon, False)
    written = recon.get_constraints()
    recon.write_fit_in()
    with open(recon.con_file) as stream:
        lines = stream.read().split('\n')
    assert float(lines[4].split()[1]) == -100.0
    recon.read_fit_in()
    restored = recon.get_constraints()
    assert len(restored) == len(written) == 10
    for con_in, con_out in zip(written, restored):
        assert type(con_out) is type(con_in)
        assert con_out.use is False
        assert con_out.err == pytest.approx(con_in.err, rel=1.E-15)
    recon.reset_constraints()
    add_every_type(recon, True)
    recon.write_fit_in()
    recon.read_fit_in()
    assert all(con.use is True for con in recon.get_constraints())


def test_recon_zero_weight_read(recon):
    '''A constraint written with weight 0 (the only unused form before the used flag) reads back as unused with no
    error and is written again with weight 0'''
    with open(recon.con_file, 'w') as stream:
        stream.write('1\n\n7\n 6.0 0.5\n 1.5 0.0\n\n')
    recon.read_fit_in()
    restored = recon._flux_loops[0]
    assert (restored.use, restored.err) == (False, None)
    recon.write_fit_in()
    with open(recon.con_file) as stream:
        assert float(stream.read().split('\n')[4].split()[1]) == 0.0


@pytest.mark.parametrize('use', ['yes', 1, None])
def test_recon_rejects_invalid_use(recon, use):
    '''The used flag must be a bool'''
    with pytest.raises(ValueError, match='`use` must be True or False'):
        recon.add_flux_loop([6.0, 0.5], 1.5, 0.01, use=use)
    with pytest.raises(ValueError, match='`use` must be True or False'):
        recon.add_MSE([6.0, 0.5], mse_coef_sets[0], 0.1, 0.01, use=use)
    with pytest.raises(ValueError, match='`use` must be True or False'):
        recon.set_coil_currents({'PF1': 1.0, 'PF2': 1.0, 'VS': 1.0}, {'PF1': 1.0, 'PF2': 1.0, 'VS': 1.0},
                                use={'PF1': use})
    assert recon.get_constraints() == []


@pytest.mark.parametrize('err,match', [
    (0.0, 'Error must be positive'),
    (-0.01, 'Error must be positive'),
    (np.nan, 'Error must be finite'),
    (np.inf, 'Error must be finite'),
    (True, 'Error must be numeric'),
    ([0.01, 0.01], r'Error must have shape \(\)'),
])
def test_recon_rejects_invalid_error(recon, err, match):
    '''Every constraint type rejects an error that is given and is not positive and finite (an unused constraint is
    expressed only by the used flag)'''
    from OpenFUSIONToolkit.TokaMaker.reconstruction import q_con, saddle_con
    adders = [
        lambda: recon.add_flux_loop([6.0, 0.5], 1.5, err),
        lambda: recon.add_Mirnov([6.0, 0.5], [1.0, 0.0, 0.0], 0.25, err),
        lambda: recon.add_Mirnov([6.0, 0.5], [1.0, 0.0, 0.0], 0.25, err, length=0.5, n_avg=4),
        lambda: recon.add_pressure([6.0, 0.5], 1.0E3, err),
        lambda: recon.set_Ip(1.0E7, err),
        lambda: recon.set_DFlux(0.05, err),
        lambda: recon.add_MSE([6.0, 0.5], mse_coef_sets[0], 0.1, err),
        lambda: recon.set_coil_currents({'PF1': 1.0E3, 'PF2': 1.0E3, 'VS': 1.0E3}, {'PF1': err, 'PF2': 10.0, 'VS': 10.0}),
        lambda: saddle_con(pt1=[6.0, 0.5], pt2=[6.5, 1.0], width=0.25, val=0.125, err=err),
        lambda: q_con(type=3, loc=0.5, val=1.5, err=err),
    ]
    for add in adders:
        with pytest.raises(ValueError, match=match):
            add()
    assert recon.get_constraints() == []


def test_recon_coil_invalid_use_keeps_constraints(recon):
    '''An invalid coil used flag is rejected before the existing coil current constraints are replaced'''
    targets, errs = {'PF1': 1.0E3, 'PF2': 1.0E3, 'VS': 1.0E3}, {'PF1': 10.0, 'PF2': 10.0, 'VS': 10.0}
    recon.set_coil_currents(targets, errs)
    with pytest.raises(ValueError, match='`use` must be True or False'):
        recon.set_coil_currents(targets, errs, use={'PF2': 'no'})
    assert [con.ind for con in recon._coil_current_cons] == [0, 1, 2]


def test_recon_used_constraint_needs_error(recon):
    '''Writing a used constraint without an error is rejected'''
    recon.add_flux_loop([6.0, 0.5], 1.5, None)
    with pytest.raises(ValueError, match='needs an error'):
        recon.write_fit_in()


def test_recon_coil_use_unknown_name(recon):
    '''A used flag for a coil set without a target is rejected'''
    with pytest.raises(KeyError, match='without a coil current target'):
        recon.set_coil_currents({'PF1': 1.0, 'PF2': 1.0, 'VS': 1.0}, {'PF1': 1.0, 'PF2': 1.0, 'VS': 1.0},
                                use={'CS1': False})
