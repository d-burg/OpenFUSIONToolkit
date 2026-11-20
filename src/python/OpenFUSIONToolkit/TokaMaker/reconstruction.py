#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
'''! Functionality for performing equilibrium reconstructions using TokaMaker

@authors Chris Hansen
@date April 2024
@ingroup doxy_oft_python
'''
from warnings import warn
from .._interface import *
from ..util import oft_warning

## @cond
class tokamaker_recon_settings_cstruct(c_struct):
    _fields_ = [("fitF", c_bool),
                ("fitP", c_bool),
                ("fit_Pscale", c_bool),
                ("fit_FFPscale", c_bool),
                ("fitR0", c_bool),
                ("fitZ0", c_bool),
                ("fitCoils", c_bool),
                ("fitF0", c_bool),
                ("fixedCentering", c_bool),
                ("pm", c_bool),
                ("infile", ctypes.c_char_p),
                ("outfile", ctypes.c_char_p)]

# tokamaker_recon_run(tMaker_ptr,vacuum,settings,error_flag)
tokamaker_recon_run = ctypes_subroutine(oftpy_lib.tokamaker_recon_run,
    [c_void_p, c_bool, ctypes.POINTER(tokamaker_recon_settings_cstruct), c_int_ptr])

# tokamaker_recon_err(tMaker_ptr,vacuum,settings,error_mat,error_flag)
tokamaker_recon_err = ctypes_subroutine(oftpy_lib.tokamaker_recon_err,
    [c_void_p, c_bool, ctypes.POINTER(tokamaker_recon_settings_cstruct), c_double_ptr, c_int_ptr])

# tokamaker_recon_setup(tMaker_ptr,settings,ncons,error_flag)
tokamaker_recon_setup = ctypes_subroutine(oftpy_lib.tokamaker_recon_setup,
    [c_void_p, ctypes.POINTER(tokamaker_recon_settings_cstruct), c_int_ptr, c_int_ptr])

# tokamaker_recon_destroy(tMaker_ptr,error_flag)
tokamaker_recon_destroy = ctypes_subroutine(oftpy_lib.tokamaker_recon_destroy,
    [c_void_p, c_int_ptr])

# tokamaker_recon_locate(tMaker_ptr,loc_mat,error_flag)
tokamaker_recon_locate = ctypes_subroutine(oftpy_lib.tokamaker_recon_locate,
    [c_void_p, c_double_ptr, c_int_ptr])
## @endcond

Mirnov_con_id = 1
Ip_con_id = 2
fluxLoop_con_id = 7
dFlux_con_id = 8
Press_con_id = 9
q_con_id = 10
saddle_con_id = 11
coil_current_con_id = 12
MSE_con_id = 13
finite_Mirnov_con_id = 14
coil_total_current_con_id = 15

## Format specification for real numbers in the constraint file (a constant of the file format, not a setting): 17
## significant digits, so every value is read back exactly as it was given
CON_FILE_REAL_FORMAT = '.16E'

## @cond
def _check_finite(name, value, shape=None):
    '''Return `value` as a float array, raising `ValueError` unless it is finite (and of `shape`, if given); booleans
    are not numbers here'''
    if numpy.asarray(value).dtype == numpy.bool_:
        raise ValueError("{0} must be numeric".format(name))
    try:
        arr = numpy.asarray(value, dtype=numpy.float64)
    except (TypeError, ValueError):
        raise ValueError("{0} must be numeric".format(name))
    if (shape is not None) and (arr.shape != shape):
        raise ValueError("{0} must have shape {1}".format(name, shape))
    if not numpy.all(numpy.isfinite(arr)):
        raise ValueError("{0} must be finite".format(name))
    return arr


def _check_location(loc):
    '''Return an R-Z location as a tuple of floats, raising `ValueError` unless it is finite with R > 0'''
    loc = _check_finite('Location', loc, (2,))
    if loc[0] <= 0.0:
        raise ValueError("Location must have R > 0")
    return (float(loc[0]), float(loc[1]))


def _check_error(err):
    '''Return a constraint error as a float, raising `ValueError` unless it is finite and positive'''
    err = float(_check_finite('Error', err, ()))
    if err <= 0.0:
        raise ValueError("Error must be positive")
    return err


def _check_use(use):
    '''Return the used flag as a bool, raising `ValueError` unless it is True or False'''
    if not isinstance(use, (bool, numpy.bool_)):
        raise ValueError("`use` must be True or False")
    return bool(use)


def _file_weight(err, use):
    '''Weight written to the constraint file: 1/err for a used constraint, -1/err for an unused one (the solver
    clips weights at zero, so the row is evaluated but does not enter the fit), 0 if there is no error'''
    if err is None:
        if use:
            raise ValueError("A used constraint needs an error")
        return 0.0
    if use:
        return 1./err
    return -1./err


def _read_weight(value):
    '''Error and used flag from a weight in the constraint file (inverse of `_file_weight`)'''
    wt = float(value)
    if wt > 0.0:
        return 1./wt, True
    if wt < 0.0:
        return -1./wt, False
    return None, False
## @endcond


class Mirnov_con:
    r'''! TokaMaker equilibrium reconstruction Mirnov sensor constraint

    The modeled signal is \f$ \mathbf{B} \cdot \mathbf{n} \f$ at the sensor location (point probe, the default), or its
    mean over `n_avg` points for a finite-length probe: the points are the midpoints of `n_avg` equal parts of a
    straight segment of length \f$ |L| \f$ centered on the sensor, at offsets
    \f$ (k - (n_{avg}-1)/2) |L| / n_{avg} \f$, \f$ k = 0 \ldots n_{avg}-1 \f$, along the unit vector
    \f$ (n_R, n_Z)/|(n_R, n_Z)| \f$ for \f$ L > 0 \f$ and along \f$ (-n_Z, n_R)/|(n_R, n_Z)| \f$ (that vector
    rotated by +90 degrees in the R-Z plane) for \f$ L < 0 \f$. This is the convention of the EFIT probe length
    `SMP2` (\f$ L < 0 \f$ for probes whose segment is perpendicular to the measured direction).
    '''
    def __init__(self, loc=None, phi=0., norm=None, val=None, err=None, length=None, n_avg=None, use=True):
        r'''! Create Mirnov sensor constraint

        @param loc Location of Mirnov in R-Z plane [2]
        @param phi Toroidal location [rad] (only meaningful with 3D fields)
        @param norm Unit normal \f$ (R,\phi,Z) \f$ [3]
        @param val Value of constraint
        @param err Error in constraint (positive)
        @param length Signed probe length \f$ L \f$ [m] (`None` for a point probe; see class description)
        @param n_avg Number of averaging points of a finite-length probe (required with `length`)
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        if norm is not None and numpy.shape(norm) != (3,):
            raise ValueError("Mirnov normal must have three components (R, phi, Z)")
        if (length is None) != (n_avg is None):
            raise ValueError("Finite-length Mirnov sensors need both `length` and `n_avg`")
        if length is not None:
            length = float(_check_finite('Mirnov length', length, ()))
            if isinstance(n_avg, (bool, numpy.bool_)) or (not isinstance(n_avg, (int, numpy.integer))) or (n_avg < 1):
                raise ValueError("Mirnov `n_avg` must be an integer >= 1")
            n_avg = int(n_avg)
            if norm is None:
                raise ValueError("Finite-length Mirnov sensors need a normal")
            norm_chk = _check_finite('Mirnov normal', norm, (3,))
            if (length != 0.0) and (norm_chk[0] == 0.0) and (norm_chk[2] == 0.0):
                raise ValueError("Finite-length Mirnov sensors need a nonzero poloidal (R, Z) normal component")
        self.loc = loc
        self.phi = phi
        self.norm = norm
        self.val = val
        self.err = None if err is None else _check_error(err)
        ## Use the constraint in the fit? (an unused constraint is evaluated, with zero weight)
        self.use = _check_use(use)
        ## Signed probe length [m] (`None` for a point probe)
        self.length = length
        ## Number of averaging points of a finite-length probe
        self.n_avg = n_avg

    def read(self, file, finite_length=False):
        '''! Read Mirnov constraint from file

        @param file Open file object containing constraint, must be positioned at start of constraint
        @param finite_length Read a finite-length probe record (type 14) instead of a point probe record (type 1)
        '''
        values = file.readline().split()
        self.loc = (float(values[0]), float(values[1]))
        self.phi = float(values[2])
        values = file.readline().split()
        self.norm = (float(values[0]), float(values[1]), float(values[2]))
        if finite_length:
            values = file.readline().split()
            self.length = float(values[0])
            self.n_avg = int(values[1])
            if self.n_avg < 1:
                raise ValueError("Invalid value in file: Mirnov `n_avg` must be >= 1")
        else:
            self.length = None
            self.n_avg = None
        values = file.readline().split()
        self.val = float(values[0])
        self.err, self.use = _read_weight(values[1])

    def write(self, file):
        '''! Write Mirnov constraint to file

        @param file Open file object for saving constraints
        '''
        if self.length is None:
            file.write('{0}\n'.format(Mirnov_con_id))
        else:
            file.write('{0}\n'.format(finite_Mirnov_con_id))
        file.write(' {0:{f}} {1:{f}} {2:{f}}\n'.format(self.loc[0], self.loc[1], self.phi, f=CON_FILE_REAL_FORMAT))
        file.write(' {0:{f}} {1:{f}} {2:{f}}\n'.format(self.norm[0], self.norm[1], self.norm[2], f=CON_FILE_REAL_FORMAT))
        if self.length is not None:
            file.write(' {0:{f}} {1:d}\n'.format(self.length, self.n_avg, f=CON_FILE_REAL_FORMAT))
        file.write(' {0:{f}} {1:{f}}\n\n'.format(self.val, _file_weight(self.err, self.use), f=CON_FILE_REAL_FORMAT))


class Ip_con:
    '''! TokaMaker equilibrium reconstruction plasma current constraint'''
    def __init__(self, val=None, err=None, use=True):
        '''! Create plasma current constraint

        @param val Value of constraint
        @param err Error in constraint
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        if val is not None and val <= 0.0:
            raise ValueError("Plasma current constraint must be positive")
        self.val = val
        self.err = None if err is None else _check_error(err)
        ## Use the constraint in the fit? (an unused constraint is evaluated, with zero weight)
        self.use = _check_use(use)

    def read(self, file):
        '''! Read plasma current constraint from file

        @param file Open file object containing constraint, must be positioned at start of constraint
        '''
        values = file.readline().split()
        Ip = float(values[0])
        if Ip <= 0.0:
            raise ValueError("Invalid value in file: Plasma current constraint must be positive")
        self.val = Ip
        self.err, self.use = _read_weight(values[1])

    def write(self, file):
        '''! Write plasma current constraint to file

        @param file Open file object for saving constraints
        '''
        file.write('{0}\n'.format(Ip_con_id))
        file.write(' {0:{f}} {1:{f}}\n\n'.format(self.val, _file_weight(self.err, self.use), f=CON_FILE_REAL_FORMAT))


class fluxLoop_con:
    r'''! TokaMaker equilibrium reconstruction full flux loop constraint

    The modeled signal is \f$ 2\pi\psi(R,Z) \f$ [Wb], with \f$ \psi \f$ TokaMaker's poloidal flux per radian (as returned
    by `get_field_eval('PSI')`). A flux in the COCOS 11 convention equals \f$ -2\pi\psi \f$, so it enters with the
    opposite sign (see \ref doc_tokamaker_recon_constraints).
    '''
    def __init__(self, loc=None, val=None, err=None, use=True):
        '''! Create full flux loop sensor constraint

        @param loc Location of Mirnov in R-Z plane [2]
        @param val Value of constraint
        @param err Error in constraint
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        self.loc = loc
        self.val = val
        self.err = None if err is None else _check_error(err)
        ## Use the constraint in the fit? (an unused constraint is evaluated, with zero weight)
        self.use = _check_use(use)

    def read(self, file):
        '''! Read full flux loop constraint from file

        @param file Open file object containing constraint, must be positioned at start of constraint
        '''
        values = file.readline().split()
        self.loc = (float(values[0]), float(values[1]))
        values = file.readline().split()
        self.val = float(values[0])
        self.err, self.use = _read_weight(values[1])

    def write(self, file):
        '''! Write full flux loop constraint to file

        @param file Open file object for saving constraints
        '''
        file.write('{0}\n'.format(fluxLoop_con_id))
        file.write(' {0:{f}} {1:{f}}\n'.format(self.loc[0], self.loc[1], f=CON_FILE_REAL_FORMAT))
        file.write(' {0:{f}} {1:{f}}\n\n'.format(self.val, _file_weight(self.err, self.use), f=CON_FILE_REAL_FORMAT))


class dFlux_con:
    '''! TokaMaker equilibrium reconstruction diamagnetic flux constraint'''
    def __init__(self, val=None, err=None, use=True):
        '''! Create diamagnetic flux constraint

        @param val Value of constraint
        @param err Error in constraint
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        self.val = val
        self.err = None if err is None else _check_error(err)
        ## Use the constraint in the fit? (an unused constraint is evaluated, with zero weight)
        self.use = _check_use(use)

    def read(self, file):
        '''! Read diamagnetic flux constraint from file

        @param file Open file object containing constraint, must be positioned at start of constraint
        '''
        values = file.readline().split()
        self.val = float(values[0])
        self.err, self.use = _read_weight(values[1])

    def write(self, file):
        '''! Write diamagnetic flux constraint to file

        @param file Open file object for saving constraints
        '''
        file.write('{0}\n'.format(dFlux_con_id))
        file.write(' {0:{f}} {1:{f}}\n\n'.format(self.val, _file_weight(self.err, self.use), f=CON_FILE_REAL_FORMAT))


class Press_con:
    '''! TokaMaker equilibrium reconstruction plasma pressure constraint'''
    def __init__(self, loc=None, val=None, err=None, use=True):
        '''! Create plasma pressure constraint

        @param loc Location of measurement in R-Z plane [2]
        @param val Value of pressure constraint
        @param err Error in constraint
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        if val is not None and val <= 0.0:
            raise ValueError("Plasma pressure constraints must be positive")
        self.loc = loc
        self.val = val
        self.err = None if err is None else _check_error(err)
        ## Use the constraint in the fit? (an unused constraint is evaluated, with zero weight)
        self.use = _check_use(use)

    def read(self, file):
        '''! Read plasma pressure constraint from file

        @param file Open file object containing constraint, must be positioned at start of constraint
        '''
        values = file.readline().split()
        self.loc = (float(values[0]), float(values[1]))
        values = file.readline().split()
        pressure = float(values[0])
        if pressure <= 0.0:
            raise ValueError("Invalid value in file: Plasma pressure constraints must be positive")
        self.val = pressure
        self.err, self.use = _read_weight(values[1])

    def write(self, file):
        '''! Write plasma pressure constraint to file

        @param file Open file object for saving constraints
        '''
        file.write('{0}\n'.format(Press_con_id))
        file.write(' {0:{f}} {1:{f}}\n'.format(self.loc[0], self.loc[1], f=CON_FILE_REAL_FORMAT))
        file.write(' {0:{f}} {1:{f}}\n\n'.format(self.val, _file_weight(self.err, self.use), f=CON_FILE_REAL_FORMAT))


class q_con:
    '''! TokaMaker equilibrium reconstruction local safety factor constraint'''
    def __init__(self, type=None, loc=0., val=None, err=None, use=True):
        r'''! Create local safety factor constraint

        @param type Type of constraint
        @param loc Location of constraint in \f$ \hat{\psi} \f$
        @param val Value of constraint
        @param err Error in constraint
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        self.type = type
        self.val = val
        self.err = None if err is None else _check_error(err)
        ## Use the constraint in the fit? (an unused constraint is evaluated, with zero weight)
        self.use = _check_use(use)
        self.loc = loc

    def read(self, file):
        '''! Read local safety factor constraint from file

        @param file Open file object containing constraint, must be positioned at start of constraint
        '''
        values = file.readline().split()
        self.type = int(values[0])
        self.loc = float(values[1])
        values = file.readline().split()
        self.val = float(values[0])
        self.err, self.use = _read_weight(values[1])

    def write(self, file):
        '''! Write local safety factor constraint to file

        @param file Open file object for saving constraints
        '''
        file.write('{0}\n'.format(q_con_id))
        file.write(' {0:E} {1:{f}}\n'.format(self.type, self.loc, f=CON_FILE_REAL_FORMAT))
        file.write(' {0:{f}} {1:{f}}\n\n'.format(self.val, _file_weight(self.err, self.use), f=CON_FILE_REAL_FORMAT))


class saddle_con:
    '''! TokaMaker equilibrium reconstruction saddle flux loop constraint '''
    def __init__(self, pt1=None, pt2=None, width=None, val=None, err=None, use=True):
        '''! Create saddle flux loop sensor constraint

        @param pt1 Location of first toroidal saddle leg in R-Z plane [2]
        @param pt2 Location of second toroidal saddle leg in R-Z plane [2]
        @param width Toroidal extent in radians
        @param val Value of saddle loop constraint
        @param err Error in constraint
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        self.pt1 = pt1
        self.pt2 = pt2
        self.width = width
        self.val = val
        self.err = None if err is None else _check_error(err)
        ## Use the constraint in the fit? (an unused constraint is evaluated, with zero weight)
        self.use = _check_use(use)

    def read(self, file):
        '''! Read saddle flux loop constraint from file

        @param file Open file object containing constraint, must be positioned at start of constraint
        '''
        values = file.readline().split()
        self.pt1 = (float(values[0]), float(values[1]))
        values = file.readline().split()
        self.pt2 = (float(values[0]), float(values[1]))
        value = file.readline()
        self.width = float(value)
        values = file.readline().split()
        self.val = float(values[0])
        self.err, self.use = _read_weight(values[1])

    def write(self, file):
        '''! Write saddle flux loop constraint to file

        @param file Open file object for saving constraints
        '''
        file.write('{0}\n'.format(saddle_con_id))
        file.write(' {0:{f}} {1:{f}}\n'.format(self.pt1[0], self.pt1[1], f=CON_FILE_REAL_FORMAT))
        file.write(' {0:{f}} {1:{f}}\n'.format(self.pt2[0], self.pt2[1], f=CON_FILE_REAL_FORMAT))
        file.write(' {0:{f}}\n'.format(self.width, f=CON_FILE_REAL_FORMAT))
        file.write(' {0:{f}} {1:{f}}\n\n'.format(self.val, _file_weight(self.err, self.use), f=CON_FILE_REAL_FORMAT))


class coil_current_con:
    '''! TokaMaker equilibrium reconstruction coil current constraint

    The modeled signal is the coil set's current [A] (type 12, the default) or, with `total=True`, its total current as
    returned by `get_coil_currents()`, which includes its share of the virtual vertical stability coil (type 15).
    '''
    def __init__(self, ind=None, val=None, err=None, use=True, total=False):
        '''! Create coil current constraint

        @param ind Index of coil
        @param val Value of constraint [A]
        @param err Error in constraint [A]
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        @param total Constrain the total current, including the virtual vertical stability coil share?
        '''
        self.ind = ind
        self.val = val
        self.err = None if err is None else _check_error(err)
        ## Use the constraint in the fit? (an unused constraint is evaluated, with zero weight)
        self.use = _check_use(use)
        if not isinstance(total, (bool, numpy.bool_)):
            raise ValueError("`total` must be True or False")
        ## Constrain the total current (including the virtual vertical stability coil share)?
        self.total = bool(total)

    def read(self, file, total=False):
        '''! Read coil current constraint from file

        @param file Open file object containing constraint, must be positioned at start of constraint
        @param total Read a total current record (type 15) instead of a coil current record (type 12)
        '''
        self.total = total
        values = file.readline()
        self.ind = int(values) - 1
        values = file.readline().split()
        self.val = float(values[0])
        self.err, self.use = _read_weight(values[1])

    def write(self, file):
        '''! Write coil current constraint to file

        @param file Open file object for saving constraints
        '''
        file.write('{0}\n'.format(coil_total_current_con_id if self.total else coil_current_con_id))
        file.write(' {0}\n'.format(self.ind+1))
        file.write(' {0:{f}} {1:{f}}\n\n'.format(self.val, _file_weight(self.err, self.use), f=CON_FILE_REAL_FORMAT))


class MSE_con:
    r'''! TokaMaker equilibrium reconstruction motional Stark effect (MSE) constraint

    The modeled signal at the chord's measurement point is (IMAS coefficient order)
    \f[ \tan \gamma = \frac{c_0 B_Z + c_1 B_R + c_2 B_\phi + c_3 E_R}{c_4 B_Z + c_5 B_R + c_6 B_\phi + c_7 E_R + c_8 E_Z} \f]
    or \f$ \gamma = \arctan(\tan \gamma) \in (-\pi/2, \pi/2) \f$ [rad], with \f$ (B_R, B_\phi, B_Z) \f$ the field in
    TokaMaker's right-handed \f$ (R, \phi, Z) \f$ frame, evaluated as for Mirnov sensors (\f$ B_\phi = F(\psi)/R \f$ at
    the point, with no test of whether the point lies inside the plasma), and the electric field \f$ (E_R, E_Z) \f$
    supplied by the caller. Any map from a device's orientation (current or toroidal field direction, sign of the
    measured angle) to this frame is the caller's statement and belongs in the signs of the coefficients.

    Where the denominator changes sign, \f$ \tan \gamma \f$ diverges and \f$ \gamma \f$ jumps by \f$ \pi \f$; a
    denominator that is exactly zero gives NaN (a failed evaluation in `reconstruct()`, a warning in `eval_error()`).
    Coefficients and electric fields for which the denominator vanishes identically are rejected.
    '''
    ## Allowed values of `quantity` and their file codes
    quantities = {'tan': 1, 'angle': 2}

    def __init__(self, loc=None, coefs=None, val=None, err=None, quantity='tan', E_R=0., E_Z=0., phi=0., use=True):
        r'''! Create MSE constraint

        @param loc Location of the measurement point in the R-Z plane [2]
        @param coefs Geometry coefficients \f$ c_0 \ldots c_8 \f$ in IMAS order [9]
        @param val Measured value (\f$ \tan \gamma \f$, or \f$ \gamma \f$ [rad] if `quantity='angle'`)
        @param err Error in constraint (same units as `val`)
        @param quantity Measured quantity, `'tan'` (\f$ \tan \gamma \f$) or `'angle'` (\f$ \gamma \f$ [rad])
        @param E_R Radial electric field at the measurement point [V/m]
        @param E_Z Vertical electric field at the measurement point [V/m]
        @param phi Toroidal location [rad] (recorded only; fields are axisymmetric)
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        ## Use the constraint in the fit? (an unused constraint is evaluated, with zero weight)
        self.use = _check_use(use)
        if quantity not in MSE_con.quantities:
            raise ValueError("MSE quantity must be one of {0}".format(tuple(MSE_con.quantities)))
        self.quantity = quantity
        self.loc = None if loc is None else _check_location(loc)
        self.E_R = float(_check_finite('E_R', E_R, ()))
        self.E_Z = float(_check_finite('E_Z', E_Z, ()))
        self.coefs = None
        if coefs is not None:
            self.coefs = tuple(float(c) for c in _check_finite('MSE coefficients', coefs, (9,)))
            if (not any(self.coefs[4:7])) and (self.coefs[7]*self.E_R + self.coefs[8]*self.E_Z == 0.0):
                raise ValueError("MSE denominator is identically zero (c_4 = c_5 = c_6 = 0 and c_7 E_R + c_8 E_Z = 0)")
        self.val = None
        if val is not None:
            self.val = float(_check_finite('MSE value', val, ()))
            if (quantity == 'angle') and (abs(self.val) >= numpy.pi/2.):
                raise ValueError("MSE angle must lie in (-pi/2, pi/2)")
        self.err = None if err is None else _check_error(err)
        self.phi = float(_check_finite('phi', phi, ()))

    def read(self, file):
        '''! Read MSE constraint from file

        @param file Open file object containing constraint, must be positioned at start of constraint
        '''
        values = file.readline().split()
        self.loc = (float(values[0]), float(values[1]))
        self.phi = float(values[2])
        self.coefs = tuple(float(value) for value in file.readline().split())
        if len(self.coefs) != 9:
            raise ValueError("Invalid MSE constraint in file: nine coefficients are required")
        values = file.readline().split()
        self.E_R = float(values[0])
        self.E_Z = float(values[1])
        code = int(file.readline())
        quantity = [key for key, val in MSE_con.quantities.items() if val == code]
        if len(quantity) != 1:
            raise ValueError("Invalid MSE quantity in file")
        self.quantity = quantity[0]
        values = file.readline().split()
        self.val = float(values[0])
        self.err, self.use = _read_weight(values[1])

    def write(self, file):
        '''! Write MSE constraint to file

        @param file Open file object for saving constraints
        '''
        file.write('{0}\n'.format(MSE_con_id))
        file.write(' {0:{f}} {1:{f}} {2:{f}}\n'.format(self.loc[0], self.loc[1], self.phi, f=CON_FILE_REAL_FORMAT))
        file.write(' ' + ' '.join('{0:{f}}'.format(c, f=CON_FILE_REAL_FORMAT) for c in self.coefs) + '\n')
        file.write(' {0:{f}} {1:{f}}\n'.format(self.E_R, self.E_Z, f=CON_FILE_REAL_FORMAT))
        file.write(' {0:d}\n'.format(MSE_con.quantities[self.quantity]))
        file.write(' {0:{f}} {1:{f}}\n\n'.format(self.val, _file_weight(self.err, self.use), f=CON_FILE_REAL_FORMAT))


con_map = {
    Mirnov_con_id: Mirnov_con,
    Ip_con_id: Ip_con,
    fluxLoop_con_id: fluxLoop_con,
    dFlux_con_id: dFlux_con,
    Press_con_id: Press_con,
    q_con_id: q_con,
    saddle_con_id: saddle_con,
    coil_current_con_id: coil_current_con,
    MSE_con_id: MSE_con,
    finite_Mirnov_con_id: Mirnov_con,
    coil_total_current_con_id: coil_current_con
}


class tokamaker_recon_settings:
    r'''! TokaMaker reconstruction settings class'''
    def __init__(self):
        ## Adjust \f$ F*F' \f$ parameterization coefficients?
        self.fitF = False
        ## Adjust \f$ P' \f$ parameterization coefficients?
        self.fitP = False
        ## Adjust \f$ P' \f$ scale factor?
        self.fit_Pscale = False
        ## Adjust \f$ F*F' \f$ scale factor?
        self.fit_FFPscale = False
        ## Utilize and adjust \f$ R_0 \f$ constraint?
        self.fitR0 = False
        ## Utilize and adjust \f$ Z_0 \f$ constraint?
        self.fitZ0 = False
        ## Allow adjustment of PF coil currents?
        self.fitCoils = False
        ## Allow adjustment of TF coil current?
        self.fitF0 = False
        ## Do not update centering (initial guess for NL solve) as solve progresses
        self.fixedCentering = False
        ## Relative step size for finite difference Jacobian calculation
        self.dx = 1.E-2
        ## Minimum absolute step size for finite difference Jacobian calculation
        self.dx_min = 1.E-8
        ## Show detailed progress output?
        self.pm = False
        ## File containing constraint definitions
        self.infile = 'fit.in'
        ## File to write output information to
        self.outfile = 'fit.out'
        # Must be added last
        self._initialized = True

    def __setattr__(self, name, value):
        # Check if the attribute is being created for the first time after initialization
        if (name not in self.__dict__) and hasattr(self, '_initialized'):
            raise AttributeError(f"Cannot add new attribute '{name}' to tokamaker_recon_settings")
        super().__setattr__(name, value)

    @property
    def fitV0(self):
        warn(
            "`fitV0` is deprecated, use `fitZ0` instead. This function will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.fitZ0

    @fitV0.setter
    def fitV0(self, value):
        warn(
            "`fitV0` is deprecated, use `fitZ0` instead. This function will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        self.fitZ0 = value

    def get_c_struct(self,oft_env):
        r'''! Get C struct representation of settings for passing to TokaMaker Fortran API'''
        c_struct_instance = tokamaker_recon_settings_cstruct()
        c_struct_instance.fitF = self.fitF
        c_struct_instance.fitP = self.fitP
        c_struct_instance.fit_Pscale = self.fit_Pscale
        c_struct_instance.fit_FFPscale = self.fit_FFPscale
        c_struct_instance.fitR0 = self.fitR0
        c_struct_instance.fitZ0 = self.fitZ0
        c_struct_instance.fitCoils = self.fitCoils
        c_struct_instance.fitF0 = self.fitF0
        c_struct_instance.fixedCentering = self.fixedCentering
        c_struct_instance.pm = self.pm
        c_struct_instance.infile = oft_env.path2c(self.infile)
        c_struct_instance.outfile = oft_env.path2c(self.outfile)
        return c_struct_instance


class reconstruction():
    '''! TokaMaker equilibrium reconstruction class

    The constraint types, their modeled signals, sign conventions and file records are described in
    \\ref doc_tokamaker_recon_constraints.
    '''
    def __init__(self,tMaker_obj,in_filename='fit.in',out_filename='fit.out'):
        '''! Create equilibrium reconstruction object

        @param tMaker_obj TokaMaker object used for computing G-S equilibria
        @param in_filename Filename to use for reconstruction input
        @param out_filename Filename to use for reconstruction outputs
        '''
        ## Grad-Shafranov object for reconstruction
        self._tMaker_obj = tMaker_obj
        ## Reconstruction specific settings object
        self.settings = tokamaker_recon_settings()
        ## Plasma current constraint
        self._Ip_con = None
        ## Diamagnetic flux constraint
        self._Dflux_con = None
        ## Flux loop constraints
        self._flux_loops = []
        ## Mirnov sensor constraints
        self._mirnovs = []
        ## Saddle loop constraints
        self._saddles = []
        ## Plasma pressure constraints
        self._pressure_cons = []
        ## Coil current constraints
        self._coil_current_cons = []
        ## MSE constraints
        self._mse_cons = []
        ## Name of constraint file (input for reconstruction)
        self.con_file = in_filename
        ## Name of reconstruction output file
        self.out_file = out_filename
        ## Number of constraints in underlying Fortran object
        self._ncons = 0
        ## Constraints with evaluation points off the mesh, moved to the nearest mesh boundary node, found by the last
        ## `setup_constraints()`: one dictionary per constraint with keys `index` (row in `eval_error` and in
        ## `get_constraints()`), `constraint`, `points` (number of moved points) and `distance` (largest move [m])
        self.projected = []
        # Update settings
        self.settings.infile = self.con_file
        self.settings.outfile = self.out_file
        # Fit-specific input file settings
        self._tMaker_obj._oft_env.oft_in_groups['gs_fit_options'] = {
            'ftol': '1.E-3',
            'xtol': '1.E-3',
            'gtol': '1.E-3',
            'maxfev': '100',
            'epsfcn': '1.E-3',
            'factor': '1.0',
            'comp_var': 'F',
            'linearized_fit': 'F'
        }
        self._tMaker_obj._oft_env.update_oft_in()

    def __del__(self):
        '''! Destroy reconstruction object'''
        try:
            self.destroy_constraints()
        except Exception:
            pass
        self._tMaker_obj = None
        self.settings = None
        self._Ip_con = None
        self._Dflux_con = None
        self._flux_loops = []
        self._mirnovs = []
        self._saddles = []
        self._pressure_cons = []
        self._coil_current_cons = []
        self._mse_cons = []
        self.con_file = None
        self.out_file = None
        self._ncons = 0

    def set_Ip(self,Ip,err,use=True):
        '''! Set plasma current constraint

        @param Ip Value of plasma current constraint
        @param err Error in constraint
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        self._Ip_con = Ip_con(val=Ip, err=err, use=use)

    def set_DFlux(self,DFlux,err,use=True):
        '''! Set diamagnetic flux constraint

        @param DFlux Value of diamagnetic flux constraint
        @param err Error in constraint
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        self._Dflux_con = dFlux_con(val=DFlux, err=err, use=use)

    def set_coil_currents(self,targets,errs,use=None,total=False):
        '''! Set coil current constraints

        The constrained quantity is each coil set's current or, with `total=True`, its total current as returned by
        `get_coil_currents()` (including the share of the virtual vertical stability coil; see
        \\ref doc_tokamaker_recon_constraints for its use with `fitCoils`). Coil sets without a target are not
        constrained. All input is checked before the existing coil current constraints are replaced.

        @param targets Dictionary of coil current targets [A] by coil set name (any subset of the coil sets)
        @param errs Dictionary of coil current constraint errors [A] (one for every target; others are ignored)
        @param use Dictionary of used flags by coil set (default: all used); an unused constraint is evaluated but has
        zero weight
        @param total Constrain total currents (True) or coil currents (False, the default), for all targets
        '''
        if use is None:
            use = {}
        if not isinstance(total, (bool, numpy.bool_)):
            raise ValueError("`total` must be True or False")
        for name in targets:
            if name not in self._tMaker_obj.coil_sets:
                raise KeyError(f"Unknown coil set '{name}'")
            if name not in errs:
                raise KeyError(f"Missing coil current error for coil set '{name}'")
            _check_finite(f"Coil current target for '{name}'", targets[name], ())
            _check_error(errs[name])
        for name in use:
            if name not in targets:
                raise KeyError(f"Used flag given for coil set '{name}' without a coil current target")
            _check_use(use[name])
        self._coil_current_cons = []
        for name, coil_set in self._tMaker_obj.coil_sets.items():
            if name not in targets:
                continue
            self._coil_current_cons.append(coil_current_con(ind=coil_set['id'], val=targets[name], err=errs[name],
                                                            use=use.get(name, True), total=total))

    def add_flux_loop(self,loc,val,err,use=True):
        r'''! Add full poloidal flux loop constraint

        @param loc Location of flux loop in R-Z plane [2]
        @param val Value of flux loop constraint [Wb], in TokaMaker's sign convention (modeled as \f$ 2\pi\psi \f$; a
        COCOS 11 flux enters with the opposite sign, see \ref doc_tokamaker_recon_constraints)
        @param err Error in constraint
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        self._flux_loops.append(fluxLoop_con(loc=loc, val=val, err=err, use=use))

    def add_Mirnov(self,loc,norm,val,err,length=None,n_avg=None,use=True):
        r'''! Add Mirnov sensor constraint

        @param loc Location of Mirnov in R-Z plane [2]
        @param norm Unit normal \f$ (R,\phi,Z) \f$ [3]; use a zero toroidal component for a poloidal probe
        @param val Value of Mirnov constraint
        @param err Error in constraint
        @param length Signed probe length [m] for a finite-length probe (`None`, the default, for a point probe; the
        segment geometry is described in \ref OpenFUSIONToolkit.TokaMaker.reconstruction.Mirnov_con "Mirnov_con")
        @param n_avg Number of averaging points of a finite-length probe (required with `length`)
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        self._mirnovs.append(Mirnov_con(loc=loc, norm=norm, val=val, err=err, length=length, n_avg=n_avg, use=use))

    def add_saddle(self,p1,p2,width,val,err,use=True):
        '''! Add saddle loop constraint

        @param p1 Location of first toroidal saddle leg in R-Z plane [2]
        @param p2 Location of second toroidal saddle leg in R-Z plane [2]
        @param width Toroidal extent in radians
        @param val Value of saddle loop constraint
        @param err Error in constraint
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        self._saddles.append(saddle_con(pt1=p1, pt2=p2, width=width, val=val, err=err, use=use))

    def add_pressure(self,loc,val,err,use=True):
        '''! Add plasma pressure constraint

        @param loc Location of measurement in R-Z plane [2]
        @param val Value of pressure constraint
        @param err Error in constraint
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        self._pressure_cons.append(Press_con(loc=loc,val=val,err=err,use=use))

    def add_MSE(self,loc,coefs,val,err,quantity='tan',E_R=0.,E_Z=0.,phi=0.,use=True):
        r'''! Add motional Stark effect (MSE) constraint (see \ref OpenFUSIONToolkit.TokaMaker.reconstruction.MSE_con
        "MSE_con" for the model and sign conventions)

        @param loc Location of the measurement point in the R-Z plane [2]
        @param coefs Geometry coefficients \f$ c_0 \ldots c_8 \f$ in IMAS order [9]
        @param val Measured value (\f$ \tan \gamma \f$, or \f$ \gamma \f$ [rad] if `quantity='angle'`)
        @param err Error in constraint (same units as `val`)
        @param quantity Measured quantity, `'tan'` (\f$ \tan \gamma \f$) or `'angle'` (\f$ \gamma \f$ [rad])
        @param E_R Radial electric field at the measurement point [V/m]
        @param E_Z Vertical electric field at the measurement point [V/m]
        @param phi Toroidal location [rad] (recorded only; fields are axisymmetric)
        @param use Use the constraint in the fit? (if False it is evaluated but has zero weight)
        '''
        for name, value in (('Location', loc), ('MSE coefficients', coefs), ('MSE value', val), ('Error', err)):
            if value is None:
                raise ValueError("{0} is required".format(name))
        self._mse_cons.append(MSE_con(loc=loc, coefs=coefs, val=val, err=err, quantity=quantity, E_R=E_R, E_Z=E_Z, phi=phi,
                                      use=use))

    def reset_constraints(self):
        '''! Remove all current constraints'''
        self._Ip_con = None
        self._Dflux_con = None
        self._flux_loops = []
        self._mirnovs = []
        self._saddles = []
        self._pressure_cons = []
        self._coil_current_cons = []
        self._mse_cons = []

    def get_constraints(self):
        '''! Get all constraints in the order of the constraint file and of the rows of `eval_error`

        The order is: flux loops, Mirnov sensors, saddle loops, pressures, coil currents, plasma current, diamagnetic
        flux, MSE.

        @result List of constraint objects
        '''
        constraints = self._flux_loops + self._mirnovs + self._saddles + self._pressure_cons + self._coil_current_cons
        if self._Ip_con is not None:
            constraints.append(self._Ip_con)
        if self._Dflux_con is not None:
            constraints.append(self._Dflux_con)
        constraints += self._mse_cons
        return constraints

    def write_fit_in(self):
        '''! Create reconstruction input file for specified constraints'''
        constraints = self.get_constraints()
        ncons = len(constraints)
        with open(self.con_file, 'w+') as fid:
            fid.write('{0:d}\n\n'.format(ncons))
            for con in constraints:
                con.write(fid)

    def read_fit_in(self):
        '''! Read constraints from existing input file'''
        self.reset_constraints()
        with open(self.con_file, 'r') as fid:
            ncons = int(fid.readline())
            for _ in range(ncons):
                fid.readline()
                con_type = int(fid.readline())
                new_con_class = con_map[con_type]
                new_con = new_con_class()
                if con_type == finite_Mirnov_con_id:
                    new_con.read(fid, finite_length=True)
                elif con_type == coil_total_current_con_id:
                    new_con.read(fid, total=True)
                else:
                    new_con.read(fid)
                if con_type == Ip_con_id:
                    self._Ip_con = new_con
                elif con_type == dFlux_con_id:
                    self._Dflux_con = new_con
                elif con_type == fluxLoop_con_id:
                    self._flux_loops.append(new_con)
                elif con_type in (Mirnov_con_id, finite_Mirnov_con_id):
                    self._mirnovs.append(new_con)
                elif con_type == saddle_con_id:
                    self._saddles.append(new_con)
                elif con_type == Press_con_id:
                    self._pressure_cons.append(new_con)
                elif con_type in (coil_current_con_id, coil_total_current_con_id):
                    self._coil_current_cons.append(new_con)
                elif con_type == MSE_con_id:
                    self._mse_cons.append(new_con)
                else:
                    raise ValueError("Unknown constraint type")

    def plot_constraints(self, fig, ax, plot_flux_loops=True, plot_mirnovs=True, plot_saddles=True, plot_pressure=True, mirnov_scale=1.0, base_zorder=0):
        '''! Plot constraint locations in R-Z plane for current constraints

        @param fig Matplotlib figure to plot on
        @param ax Matplotlib axis to plot on
        @param plot_flux_loops Whether to plot flux loop constraints
        @param plot_mirnovs Whether to plot Mirnov sensor constraints
        @param plot_saddles Whether to plot saddle loop constraints
        @param plot_pressure Whether to plot plasma pressure constraints
        @param mirnov_scale Scaling factor for Mirnov sensor quiver plot (default 1.0, smaller values increase arrow size)
        @param base_zorder Base z-order for plotting constraints (default 0.0, higher values will plot on top of lower values)
        '''
        if plot_flux_loops:
            for flux_loop in self._flux_loops:
                ax.plot(flux_loop.loc[0], flux_loop.loc[1], 'ro', zorder=base_zorder+1)
        mirnov_scale = 1.0/mirnov_scale/(0.005*(self._tMaker_obj.r.max(axis=0) - self._tMaker_obj.r.min(axis=0)).max())
        if plot_mirnovs:
            for mirnov in self._mirnovs:
                ax.quiver(mirnov.loc[0], mirnov.loc[1], mirnov.norm[0], mirnov.norm[2], color='g', scale=mirnov_scale, zorder=base_zorder+2)
        if plot_saddles:
            for saddle in self._saddles:
                ax.plot([saddle.pt1[0],saddle.pt2[0]], [saddle.pt1[1],saddle.pt2[1]], 'b+', zorder=base_zorder+1)
        if plot_pressure:
            for press_con in self._pressure_cons:
                ax.plot(press_con.loc[0], press_con.loc[1], 'm.', zorder=base_zorder+1)

    def plot_error(self, ax, error_mat=None, error_file=None, coil_ax=None, chi_ax=None):
        '''! Plot constraint values and reconstructed signals for current equilibrium from reconstruction output file

        @param ax Matplotlib axis to plot on
        @param error_mat Error matrix, if `None` `eval_error` will be called
        @param error_file File containing error data
        @param coil_ax Matplotlib axis to plot coil current constraints on
        @param chi_ax Matplotlib axis to plot chi-squared values on
        '''
        if len(ax) != 4:
            raise ValueError("Must provide list of 4 axes for plotting flux loop, Mirnov, saddle, and pressure constraint errors")
        if error_file is not None:
            error_mat = numpy.loadtxt(error_file)[:,1:]
        if error_mat is None:
            error_mat = self.eval_error()
        else:
            if error_mat.shape[1] != 4:
                raise ValueError("Error matrix must have 4 columns (weighted error, reconstructed signal, constraint value, 3D correction)")
            if error_mat.shape[0] != self._ncons:
                raise ValueError("Error matrix must have same number of rows as number of constraints in reconstruction")
        # Plot signals for "main" diagnostics
        i = 0
        ax[0].set_title('Flux Loop Constraints')
        ax[0].set_xlabel('Constraint Index')
        ax[0].set_ylabel('Signal [Wb]')
        for j, flux_loop in enumerate(self._flux_loops):
            ax[0].errorbar(j,error_mat[i,2], yerr=flux_loop.err, color='r', capsize=2)
            ax[0].plot(j,error_mat[i,1], 'bx')
            i += 1
        ax[1].set_title('Mirnov Constraints')
        ax[1].set_xlabel('Constraint Index')
        ax[1].set_ylabel('Signal [T]')
        for j, mirnov in enumerate(self._mirnovs):
            ax[1].errorbar(j,error_mat[i,2], yerr=mirnov.err, color='r', capsize=2)
            ax[1].plot(j,error_mat[i,1], 'bx')
            i += 1
        ax[2].set_title('Saddle Constraints')
        ax[2].set_xlabel('Constraint Index')
        ax[2].set_ylabel('Signal [Wb]')
        for j, saddle in enumerate(self._saddles):
            ax[2].errorbar(j,error_mat[i,2], yerr=saddle.err, color='r', capsize=2)
            ax[2].plot(j,error_mat[i,1], 'bx')
            i += 1
        ax[3].set_title('Pressure Constraints')
        ax[3].set_xlabel('Constraint Index')
        ax[3].set_ylabel('Signal [Pa]')
        for j, press_con in enumerate(self._pressure_cons):
            ax[3].errorbar(j,error_mat[i,2], yerr=press_con.err, color='r', capsize=2)
            ax[3].plot(j,error_mat[i,1], 'bx')
            i += 1
        # Plot coil signals
        if (coil_ax is not None) and (len(self._coil_current_cons) > 0):
            ind_to_name = {self._tMaker_obj.coil_sets[coil_set]['id']: coil_set for coil_set in self._tMaker_obj.coil_sets}
            coil_ax.set_title('Coil Current Constraints')
            coil_ax.set_xlabel('Coil Set Index')
            coil_ax.set_ylabel('Current [A]')
            for coil_con in self._coil_current_cons:
                coil_ax.errorbar(coil_con.ind, coil_con.val, yerr=coil_con.err, color='r', capsize=2)
                coil_ax.plot(coil_con.ind, error_mat[i,1], 'bx')
                i += 1
            coil_ax.set_xticks(range(len(ind_to_name)), labels=[ind_to_name[j] for j in sorted(ind_to_name.keys())],
                rotation=45, ha="right", rotation_mode="anchor")
        # Plot error contributions
        if chi_ax is not None:
            err_ind = numpy.cumsum([0,
                                    len(self._flux_loops),
                                    len(self._mirnovs),
                                    len(self._saddles),
                                    len(self._pressure_cons),
                                    len(self._coil_current_cons),
                                    1 if self._Ip_con is not None else 0,
                                    1 if self._Dflux_con is not None else 0,
                                    len(self._mse_cons)])
            names = ['Flux loops', 'Mirnovs', 'Saddle loops', 'Pressure', r'$I_C$', r'$I_p$', r'$\Delta \Phi$', 'MSE']
            chi_ax.set_title(r'Signal $\chi^2$ contributions')
            chi_ax.set_xlabel('Signal Index')
            chi_ax.set_ylabel(r'$\chi^2_i$')
            for i in range(err_ind.shape[0]-1):
                if err_ind[i+1] == err_ind[i]:
                    continue
                chi_ax.plot(numpy.arange(err_ind[i],err_ind[i+1])+1,numpy.power(error_mat[err_ind[i]:err_ind[i+1],0],2),label=names[i])
            chi_ax.legend()

    def setup_constraints(self):
        '''! Set up constraints in TokaMaker for current equilibrium without performing reconstruction

        Evaluation points that lie off the mesh are moved to the nearest mesh boundary node, as in `reconstruct()`;
        such constraints are listed in `projected` and a warning is printed.

        @result Error flag
        '''
        if self._ncons > 0:
            self.destroy_constraints()
        # Modify input file
        self.write_fit_in()
        # Run setup
        ncons = c_int()
        error_flag = c_int()
        tokamaker_recon_setup(self._tMaker_obj._tMaker_ptr,ctypes.byref(self.settings.get_c_struct(self._tMaker_obj._oft_env)),ctypes.byref(ncons),ctypes.byref(error_flag))
        if error_flag.value != 0:
            raise ValueError("Constraint setup failed with error code {0:d}".format(error_flag.value))
        self._ncons = ncons.value
        # Locate evaluation points and report those moved onto the mesh boundary
        loc_mat = numpy.zeros((self._ncons, 2), dtype=numpy.float64)
        tokamaker_recon_locate(self._tMaker_obj._tMaker_ptr,loc_mat.ctypes.data_as(c_double_ptr),ctypes.byref(error_flag))
        if error_flag.value != 0:
            raise ValueError("Constraint location failed with error code {0:d}".format(error_flag.value))
        self.projected = []
        for i, con in enumerate(self.get_constraints()):
            if loc_mat[i,0] > 0.0:
                self.projected.append({'index': i, 'constraint': con, 'points': int(loc_mat[i,0]), 'distance': loc_mat[i,1]})
        if len(self.projected) > 0:
            oft_warning('{0:d} constraint(s) have evaluation points off the mesh, moved to the nearest boundary node (largest move {1:.3E} m); see `projected`'.format(
                len(self.projected), max([entry['distance'] for entry in self.projected])))

    def destroy_constraints(self):
        '''! Destroy constraints in TokaMaker'''
        if self._ncons == 0:
            return
        error_flag = c_int()
        tokamaker_recon_destroy(self._tMaker_obj._tMaker_ptr,ctypes.byref(error_flag))
        if error_flag.value != 0:
            raise ValueError("Constraint destruction failed with error code {0:d}".format(error_flag.value))
        self._ncons = 0

    def eval_error(self, vacuum=False, save_to_file=False):
        '''! Evaluate error in current equilibrium for specified constraints without performing reconstruction

        @param vacuum Perform vacuum reconstruction
        @param save_to_file Save error matrix to file specified in `settings.outfile`
        @result Error matrix if `save_to_file=False`, otherwise None (error matrix will be saved to file)
        '''
        if self._ncons == 0:
            raise ValueError("Constraints are not set up; call `setup_constraints()` before `eval_error()`. ")
        mat_ptr = None
        if not save_to_file:
            error_mat = numpy.zeros((self._ncons, 4), dtype=numpy.float64)
            mat_ptr = error_mat.ctypes.data_as(c_double_ptr)
        error_flag = c_int()
        tokamaker_recon_err(self._tMaker_obj._tMaker_ptr,c_bool(vacuum),ctypes.byref(self.settings.get_c_struct(self._tMaker_obj._oft_env)),mat_ptr,ctypes.byref(error_flag))
        if error_flag.value != 0:
            raise ValueError("Error evaluation failed with error code {0:d}".format(error_flag.value))
        if save_to_file:
            return None
        bad_rows = numpy.flatnonzero(numpy.logical_not(numpy.isfinite(error_mat[:,1])))
        if bad_rows.shape[0] > 0:
            oft_warning('Modeled signal is not finite for constraint row(s) {0} (e.g. an MSE denominator of zero)'.format(
                bad_rows.tolist()))
        return error_mat

    def setup_get_opt(self,fail_factor=1.E2,pp_target_weight=1.E5,opoint_target_weight=1.E2):
        r'''! Setup object for use with general optimization routines, such as `scipy.optimize.minimize`,
        which will call `opt_error` and, optionally, `opt_error_jacobian` with appropriate arguments.
        This function should be called after setting up constraints with `setup_constraints` and before calling optimization routines.

        @param fail_factor Factor to multiply initial error by for failed solves
        @param pp_target_weight Weight for global P' targets when treated as soft constraints and fit_Pscale is True
        @param opoint_target_weight Weight for R0 and Z0 targets when treated as soft constraints and `fitR0` or `fitZ0` is True
        @returns Initial DoF vector for optimization, error vector for initial DoFs
        '''
        x0_nl = self.opt_get_dofs()
        self._err_min = 1.E99
        self._EQ_center = self._tMaker_obj.copy_eq()
        error0  = self.opt_error(x0_nl,self,False)
        self._fail_error = error0*fail_factor

        # Make sure settings are up to date
        soft_targets_active = any([self._tMaker_obj.settings.ffp_target_weight > 0.0, self._tMaker_obj.settings.pp_target_weight > 0.0, self._tMaker_obj.settings.opoint_target_weight > 0.0])
        if (self.settings.fitR0 or self.settings.fitZ0) and (soft_targets_active):
            self._tMaker_obj.settings.opoint_target_weight=opoint_target_weight
        if self.settings.fit_Pscale and soft_targets_active:
            self._tMaker_obj.settings.pp_target_weight=pp_target_weight
        self._tMaker_obj.update_settings()

        return x0_nl, error0

    def opt_get_dofs(self):
        r'''! Get DoF vector for general optimization routines using `opt_error` and `opt_error_jacobian`.

        @returns DoF vector
        '''
        dofs = []
        if self.settings.fitR0:
            dofs.append(self._tMaker_obj.o_point[0])
        if self.settings.fitZ0:
            dofs.append(self._tMaker_obj.o_point[1])
        if self.settings.fit_Pscale:
            dofs.append(self._tMaker_obj.p_scale)
        if self.settings.fitF:
            f_dofs = self._tMaker_obj.get_profile_dofs('ffp')
            if f_dofs is not None:
                dofs.extend(f_dofs.tolist())
        if self.settings.fitP:
            p_dofs = self._tMaker_obj.get_profile_dofs('pp')
            if p_dofs is not None:
                dofs.extend(p_dofs.tolist())
        return numpy.array(dofs)

    @staticmethod
    def opt_error(cofs,recon_obj,in_jac):
        r'''! Compute weighted constraint error for given DoF vector `cofs` and reconstruction object `recon_obj`.
        This function is designed to be called by general optimization routines, such as `scipy.optimize.minimize`.

        @param cofs DoF vector
        @param recon_obj `TokaMaker.reconstruction.reconstruction` object to be used
        @param in_jac Flag indicating whether this function is being called from a Jacobian calculation to make print statements during finite differencing
        @returns Weighted error vector for current DoFs
        '''
        offset = 0
        if recon_obj.settings.fitR0:
            recon_obj._tMaker_obj.set_targets(R0=cofs[offset],retain_previous=True)
            offset += 1
        if recon_obj.settings.fitZ0:
            recon_obj._tMaker_obj.set_targets(Z0=cofs[offset],retain_previous=True)
            offset += 1
        if recon_obj.settings.fit_Pscale:
            recon_obj._tMaker_obj.p_scale=cofs[offset]
            offset += 1
        if recon_obj.settings.fitF:
            f_dofs = recon_obj._tMaker_obj.get_profile_dofs('ffp')
            if f_dofs is not None:
                f_dofs = cofs[offset:offset+f_dofs.shape[0]]
                recon_obj._tMaker_obj.set_profile_dofs('ffp', f_dofs)
                offset += f_dofs.shape[0]
        if recon_obj.settings.fitP:
            p_dofs = recon_obj._tMaker_obj.get_profile_dofs('pp')
            if p_dofs is not None:
                p_dofs = cofs[offset:offset+p_dofs.shape[0]]
                recon_obj._tMaker_obj.set_profile_dofs('pp', p_dofs)
                offset += p_dofs.shape[0]

        # Re-solve
        try:
            recon_obj._tMaker_obj.solve()
            if recon_obj._tMaker_obj.alam == 0.0:
                recon_obj._tMaker_obj.replace_eq(recon_obj._EQ_center)
                return recon_obj._fail_error
        except:
            recon_obj._tMaker_obj.replace_eq(recon_obj._EQ_center)
            return recon_obj._fail_error

        # Compute error
        err_out = recon_obj.eval_error()
        total_err = numpy.power(numpy.linalg.norm(err_out[:,0]),2)
        if not in_jac:
            print('chi_rms = {0:.4E}; chi_max = {1:.4E}'.format(numpy.sqrt(total_err/err_out.shape[0]), abs(err_out[:,0]).max()))
            if total_err < recon_obj._err_min:
                if not recon_obj.settings.fixedCentering:
                    recon_obj._EQ_center = recon_obj._tMaker_obj.copy_eq()
                recon_obj._err_min = total_err
        return err_out[:,0]


    @staticmethod
    def opt_error_jacobian(cofs,recon_obj,in_jac):
        r'''! Compute Jacobian matrix corresponding by differencing `opt_error` for given DoF vector `cofs` and reconstruction object `recon_obj`.

        @param cofs DoF vector
        @param recon_obj `TokaMaker.reconstruction.reconstruction` object to be used
        @param in_jac Ignored flag just to match signature with `opt_error`
        @returns Jacobian matrix for current DoFs
        '''
        recon_obj._tMaker_obj.replace_eq(recon_obj._EQ_center)
        _ = reconstruction.opt_error(cofs,recon_obj,True)
        jac = numpy.zeros((len(recon_obj._fail_error),len(cofs)))
        center_EQ = recon_obj._tMaker_obj.copy_eq()
        # Resolve for center point to capture "one more step" converged result as achieved in small diffs below
        center_err = reconstruction.opt_error(cofs,recon_obj,True)
        for i, cof_val in enumerate(cofs):
            cof_tmp = cofs.copy()
            cof_diff = max(recon_obj.settings.dx_min,recon_obj.settings.dx*abs(cof_val))
            cof_tmp[i] += cof_diff
            pert_err = reconstruction.opt_error(cof_tmp,recon_obj,True)
            jac[:,i] = (pert_err-center_err)/cof_diff
            recon_obj._tMaker_obj.replace_eq(center_EQ)
        return jac

    def reconstruct(self, vacuum=False, linearized_fit=False, maxits=100, eps=1.E-3, ftol=1.E-3, xtol=1.E-3, gtol=1.E-3):
        '''! Reconstruct G-S equation with specified fitting constraints, profiles, etc.

        @param vacuum Perform vacuum reconstruction
        @param linearized_fit Use linearized solve for suitable terms
        @param maxits Maximum number of iterations
        @param eps Epsilong factor for finite difference derivative calculations
        @param ftol Stopping condition: termination occurs when both the actual and predicted relative reductions in the sum of squares are at most `ftol`
        @param xtol Stopping condition: termination occurs when the relative error between two consecutive iterates is at most `xtol`
        @param gtol Stopping condition: termination occurs when the cosine of the angle between fvec and any column of the jacobian is at most `gtol` in absolute value
        @result Error flag
        '''
        # Check for possibly conflicting constraints
        if self._tMaker_obj._tMaker_equil.Isoflux_constraints is not None:
            oft_warning('Removing conflicting isoflux constraints from equilibrium object via `.set_isoflux_constraints(None)`')
            self._tMaker_obj.set_isoflux_constraints(None)
        if self._tMaker_obj._tMaker_equil.Psi_constraints[0] is not None:
            oft_warning('Removing conflicting Psi constraints from equilibrium object via `.set_psi_constraints(None,None)`')
            self._tMaker_obj.set_psi_constraints(None,None)
        if self._tMaker_obj._tMaker_equil.Saddle_constraints is not None:
            oft_warning('Removing conflicting saddle targets from equilibrium object via `.set_saddle_constraints(None)`')
            self._tMaker_obj.set_saddle_constraints(None)
        weight_chk = [self._tMaker_obj.settings.ffp_target_weight > 0.0, self._tMaker_obj.settings.pp_target_weight > 0.0, self._tMaker_obj.settings.opoint_target_weight > 0.0]
        if any(weight_chk):
            oft_warning('Removing conflicting soft targets from equilibrium solve via `.settings.*_target_weight = -1.0; .update_settings()`')
            self._tMaker_obj.settings.ffp_target_weight = -1.0
            self._tMaker_obj.settings.pp_target_weight = -1.0
            self._tMaker_obj.settings.opoint_target_weight = -1.0
            self._tMaker_obj.update_settings()
        # Modify input file
        self.write_fit_in()
        self._tMaker_obj._oft_env.oft_in_groups['gs_fit_options']['linearized_fit'] = 'T' if linearized_fit else 'F'
        self._tMaker_obj._oft_env.oft_in_groups['gs_fit_options']['maxfev'] = '{0:d}'.format(maxits)
        self._tMaker_obj._oft_env.oft_in_groups['gs_fit_options']['epsfcn'] = '{0:.5E}'.format(eps)
        self._tMaker_obj._oft_env.oft_in_groups['gs_fit_options']['ftol'] = '{0:.5E}'.format(ftol)
        self._tMaker_obj._oft_env.oft_in_groups['gs_fit_options']['xtol'] = '{0:.5E}'.format(xtol)
        self._tMaker_obj._oft_env.oft_in_groups['gs_fit_options']['gtol'] = '{0:.5E}'.format(gtol)
        self._tMaker_obj._oft_env.update_oft_in()
        # Run reconstruction
        error_flag = c_int()
        tokamaker_recon_run(self._tMaker_obj._tMaker_ptr,c_bool(vacuum),ctypes.byref(self.settings.get_c_struct(self._tMaker_obj._oft_env)),ctypes.byref(error_flag))
        return error_flag.value


#------------------------------------------------------------------------------
# Kinetic equilibrium reconstruction of one time slice
#------------------------------------------------------------------------------
## Mode of @ref kinetic_reconstruction.run "kinetic_reconstruction.run()" implemented here
KINETIC_MODES = ('slice',)
## Modes of the later pull requests and what each will do
KINETIC_MODES_LATER = {
    'slices': 'many independent slices, each fitted as a single slice (a later pull request)',
    'series': 'a time-dependent series with current diffusion between the slices (a later pull request)',
}
## Keys of the device description `machine` of @ref kinetic_reconstruction (`vsc` required)
KINETIC_MACHINE_KEYS = ('name', 'vsc', 'r0')


class kinetic_slice_data():
    '''! Inputs of one time slice of a kinetic reconstruction

    The inputs are the device-independent containers of the helper package
    @ref OpenFUSIONToolkit.TokaMaker.kinetic "TokaMaker.kinetic"; they are handed to the fit unchanged (nothing is
    computed, converted or copied here). A machine module builds them from a device's own files; a user with data in
    standard form builds them directly. See \\ref doc_tokamaker_kinetic_reconstruction.
    '''
    def __init__(self, measurements, kinetics, sources, boundary, F0, start, coil_roles=None, reference_points=None,
                 initial_profiles=None, params0=None):
        r'''! Collect the inputs of one slice

        @param measurements Measurements of the slice (`kinetic.interfaces.MeasurementSet`); its `time` [s] names the slice
        @param kinetics Thermal kinetic profiles: `kinetic.kinetics.KineticInputs` (on rho_tor_norm, built by
        `kinetic.kinetics.assemble_kinetic_inputs`) or `kinetic.current_model.KineticProfiles` (on psi_N)
        @param sources Driven current and fast-ion pressure (`kinetic.current_model.SourceProfiles`; no sources:
        `SourceProfiles(psi_n)`)
        @param boundary Anchor of the forward solve: (R, Z) points [m] of a prior equilibrium's last closed flux surface,
        shape (n, 2)
        @param F0 Vacuum \f$ R B_\phi \f$ [T m], signed as the discharge
        @param start Starting state of the fit: \f$ \psi \f$ on the mesh nodes [Wb/rad] in TokaMaker's convention (for
        example `kinetic.forward.psi_on_nodes` of a prior equilibrium), or an equilibrium copy of the reconstruction's
        own solver (a previous result's @ref kinetic_slice_result.equilibrium "equilibrium()")
        @param coil_roles Coil name -> `'fitted'`, `'held'` or `'free'`; None: `measurements.groups['pf_current'].meta['coil_roles']`
        @param reference_points Points of the anchor equilibrium that are only recorded (displacement of the fitted
        boundary): `'x_points'` [(R, Z), ...] and `'strike_points'` {name: (R, Z)}
        @param initial_profiles `(x, <j_phi>, p)` on psi_N for the first solve when `start` is only a guess
        @param params0 Starting parameters of the fit (default: the prior means, `kinetic.fit.reconstruct`)
        '''
        ## Measurements of the slice (`MeasurementSet`)
        self.measurements = measurements
        ## Thermal kinetic profiles (`KineticInputs` or `KineticProfiles`)
        self.kinetics = kinetics
        ## Driven current and fast-ion pressure (`SourceProfiles`)
        self.sources = sources
        ## Anchor boundary points [m], shape (n, 2)
        self.boundary = boundary
        ## Vacuum \f$ R B_\phi \f$ [T m]
        self.F0 = F0
        ## Starting state (psi on the mesh nodes, or an equilibrium copy of the reconstruction's solver)
        self.start = start
        ## Coil roles (None: from the measurement set)
        self.coil_roles = coil_roles
        ## Recorded reference points of the anchor equilibrium
        self.reference_points = reference_points
        ## Profiles of the first solve
        self.initial_profiles = initial_profiles
        ## Starting parameters of the fit
        self.params0 = params0

    @property
    def time(self):
        '''! Time of the slice [s]: the measurement set's `time`'''
        return self.measurements.time


class kinetic_slice_result():
    '''! Result of a single-slice kinetic reconstruction (@ref kinetic_reconstruction.run "kinetic_reconstruction.run()")

    Holds the fit result of `kinetic.fit.reconstruct` unchanged (`fit`) and exposes its parts. `record` is the fit
    record: every setting, prior, stopping test, solver setting and diagnostic in effect.
    '''
    def __init__(self, time, fit):
        '''! Wrap a fit result

        @param time Time of the slice [s]
        @param fit `kinetic.fit.FitResult` returned by `kinetic.fit.reconstruct`
        '''
        ## Time of the slice [s]
        self.time = time
        ## The fit result of `kinetic.fit.reconstruct`, unchanged
        self.fit = fit

    @property
    def status(self):
        '''! Status of the fit (`converged`, `converged_within_noise`, `stalled_no_significant_descent`, ...)'''
        return self.fit.status

    @property
    def record(self):
        '''! The fit record (dict)'''
        return self.fit.record

    @property
    def params(self):
        '''! All model parameters at the fitted state (dict)'''
        return self.fit.params

    @property
    def free(self):
        '''! Names of the fitted parameters'''
        return self.fit.free

    @property
    def theta(self):
        '''! Fitted values of the free parameters'''
        return self.fit.theta

    @property
    def sigma(self):
        '''! Laplace (posterior) standard deviations of the free parameters'''
        return self.fit.sigma

    @property
    def covariance(self):
        '''! Laplace covariance of the free parameters'''
        return self.fit.covariance

    @property
    def chi2(self):
        '''! Chi-squared summary of the fit (dict)'''
        return self.fit.chi2

    def equilibrium(self):
        '''! The fitted equilibrium: a copy (`TokaMaker_equilibrium`) of the solver's state at the fit's final center

        @result Equilibrium copy, or None when the fit has no consistent final center
        '''
        center = self.fit.record.get('center')
        return None if center is None else center['snap']

    def save_geqdsk(self, filename, **kwargs):
        '''! Write the fitted equilibrium as a g-file with TokaMaker's own writer

        The file carries the full pressure: `PRES` is the solver's pressure plus the separatrix pressure of the fit's
        final center (TokaMaker's `lcfs_pressure`), `PPRIME` is unchanged (`kinetic.pressure.save_full_pressure_eqdsk`).

        @param filename Name of the g-file
        @param kwargs Passed unchanged to `TokaMaker_equilibrium.save_eqdsk` (grid, `lcfs_pad`, `cocos`, ...; its
        defaults apply); `lcfs_pressure` is refused
        @result Record of the written file (file name, separatrix pressure, pressure frame)
        '''
        from .kinetic import pressure as kinetic_pressure
        center = self.fit.record.get('center')
        if center is None:
            raise ValueError('the fit has no consistent final center (status {0!r}): no equilibrium to write'.format(self.status))
        return kinetic_pressure.save_full_pressure_eqdsk(center['snap'], filename, center['res'].solve.record['p_sep'], **kwargs)


class kinetic_reconstruction():
    '''! Kinetic equilibrium reconstruction with TokaMaker

    A MAP fit of a small set of physical parameters of a current model (ohmic, bootstrap and driven current; thermal
    and fast-ion pressure) iterated to consistency with an anchored free-boundary solve, against magnetic, coil, plasma
    current and MSE measurements. The fit itself is `kinetic.fit.reconstruct`; this class builds its forward model
    from a mesh file and a device description, passes the settings objects of the helper package through unchanged and
    adds no computation. See \\ref doc_tokamaker_kinetic_reconstruction.

    Only single slices are implemented (`run(..., mode='slice')`).
    '''
    def __init__(self, oft_env, mesh_file, machine, forward_settings=None, fit_settings=None, model_settings=None, speed=None):
        '''! Create a kinetic reconstruction

        @param oft_env OFT runtime environment (@ref OpenFUSIONToolkit._core.OFT_env "OFT_env"; one per process)
        @param mesh_file TokaMaker mesh file (HDF5, @ref OpenFUSIONToolkit.TokaMaker.meshing.save_gs_mesh "save_gs_mesh()")
        @param machine Device description: dict with `'vsc'` (coil name -> gain of the virtual vertical stability coil,
        @ref OpenFUSIONToolkit.TokaMaker.TokaMaker.set_coil_vsc "set_coil_vsc()") and optionally `'r0'` (reference
        major radius of the current model's geometry [m]) and `'name'`; a machine module supplies it
        @param forward_settings `kinetic.forward.ForwardSettings` (None: its defaults with `vsc` from `machine`)
        @param fit_settings `kinetic.fit.FitSettings` (None: its defaults)
        @param model_settings `kinetic.current_model.CurrentModelSettings` (None: its defaults with `r0` from `machine`)
        @param speed `kinetic.speed.SpeedSettings` (None: the default of `kinetic.fit.reconstruct`)
        '''
        from .kinetic import current_model as kinetic_model
        from .kinetic import fit as kinetic_fit
        from .kinetic import forward as kinetic_forward
        from .meshing import load_gs_mesh
        machine = dict(machine)
        unknown = sorted(set(machine) - set(KINETIC_MACHINE_KEYS))
        if unknown:
            raise ValueError('unknown machine description keys {0}; known: {1}'.format(unknown, list(KINETIC_MACHINE_KEYS)))
        if not machine.get('vsc'):
            raise ValueError("machine['vsc'] (coil name -> vertical stability coil gain) is required: it is device data")
        ## OFT runtime environment
        self._oft_env = oft_env
        ## Mesh file
        self.mesh_file = mesh_file
        ## Device description
        self.machine = machine
        ## Forward-solve settings (`ForwardSettings`)
        self.forward_settings = self._with_device_value(forward_settings, kinetic_forward.ForwardSettings, 'vsc', dict(machine['vsc']))
        ## Fit settings (`FitSettings`)
        self.fit_settings = kinetic_fit.FitSettings() if fit_settings is None else fit_settings
        ## Current-model settings (`CurrentModelSettings`)
        self.model_settings = self._with_device_value(model_settings, kinetic_model.CurrentModelSettings, 'r0', machine.get('r0'))
        if self.model_settings.r0 is None:
            raise ValueError("the current model's reference major radius is required: machine['r0'] or model_settings.r0")
        ## Speed settings (`SpeedSettings`, None: the fit's default)
        self.speed = speed
        self._mesh = load_gs_mesh(mesh_file)
        self._fm = None
        self._forward_settings_used = None

    @staticmethod
    def _with_device_value(settings, cls, name, value):
        '''! Settings object with a device value filled in from the device description (no other field touched)'''
        import dataclasses
        if settings is None:
            return cls() if value is None else cls(**{name: value})
        if not isinstance(settings, cls):
            raise TypeError('expected a {0}, got {1}'.format(cls.__name__, type(settings).__name__))
        given = getattr(settings, name)
        if value is None or given == value:
            return settings
        if given is None or (isinstance(given, dict) and not given):
            return dataclasses.replace(settings, **{name: value})
        raise ValueError('{0}.{1} = {2!r} disagrees with the device description ({3!r})'.format(cls.__name__, name, given, value))

    @property
    def forward_model(self):
        '''! The forward model (`kinetic.forward.ForwardModel`), set up by the first @ref kinetic_reconstruction.run "run()"; None before'''
        return self._fm

    @property
    def gs(self):
        '''! The TokaMaker object of the forward model (None before the first @ref kinetic_reconstruction.run "run()")'''
        return None if self._fm is None else self._fm.gs

    @staticmethod
    def _select(data, time):
        '''! The inputs of the slice at `time`: `data[time]` for a mapping, else `data` with `data.time == time`'''
        if isinstance(data, dict):
            if time not in data:
                raise ValueError('no inputs at time {0!r}; times given: {1}'.format(time, sorted(data)))
            data = data[time]
        if not isinstance(data, kinetic_slice_data):
            raise TypeError('data must be a kinetic_slice_data or a dict of them keyed by time')
        if data.time != time:
            raise ValueError('time {0!r} differs from the measurement time {1!r} of the inputs'.format(time, data.time))
        return data

    def _forward_model_for(self, data, label):
        '''! Set up the forward model on the first run; rebuild its rows on the same solver afterwards'''
        import copy
        from .kinetic import forward as kinetic_forward
        if self._fm is None:
            if isinstance(data.start, numpy.ndarray) and data.initial_profiles is None:
                raise ValueError('the first run starts a new solver from psi on the mesh nodes, which has no solved '
                                 'equilibrium to evaluate the current model on: give initial_profiles (x, <j_phi>, p)')
            self._fm = kinetic_forward.ForwardModel(self._mesh, data.F0, data.measurements, data.boundary,
                                                    coil_roles=data.coil_roles, settings=self.forward_settings,
                                                    oft_env=self._oft_env, case_id=label,
                                                    reference_points=data.reference_points)
            self._forward_settings_used = copy.deepcopy(self.forward_settings.as_dict())
            return self._fm
        if self.forward_settings is not self._fm.settings or self.forward_settings.as_dict() != self._forward_settings_used:
            raise RuntimeError('forward_settings changed after the solver was set up by the first run; they are read once '
                               'per solver: use a new process for other forward settings')
        self._fm.set_F0(data.F0)
        self._fm.reconfigure(data.measurements, data.boundary, data.coil_roles, case_id=label,
                             reference_points=data.reference_points)
        return self._fm

    def run(self, data, time, *, mode, label='fit', steady_state_check=True):
        '''! Reconstruct

        @param data Inputs (@ref kinetic_slice_data "kinetic_slice_data", or a dict of them keyed by time)
        @param time Time of the slice [s]; must equal the measurement time of the inputs
        @param mode `'slice'` (one slice, the only mode implemented); `'slices'` and `'series'` come with later pull
        requests. Required: the mode is never inferred from `time`
        @param label Label of the fit and of the forward model's case in the record
        @param steady_state_check Record the slice's unfitted model steady state next to the fit (a record only;
        `kinetic.fit.reconstruct`)
        @result @ref kinetic_slice_result "kinetic_slice_result"
        '''
        from .kinetic import fit as kinetic_fit
        if mode in KINETIC_MODES_LATER:
            raise NotImplementedError("mode {0!r} ({1}) is not implemented; only mode='slice' is".format(mode, KINETIC_MODES_LATER[mode]))
        if mode not in KINETIC_MODES:
            raise ValueError('mode must be one of {0}; {1} come with later pull requests'.format(list(KINETIC_MODES), list(KINETIC_MODES_LATER)))
        d = self._select(data, time)
        fm = self._forward_model_for(d, label)
        fr = kinetic_fit.reconstruct(fm, d.kinetics, d.sources, self.model_settings, settings=self.fit_settings,
                                     params0=d.params0, start=d.start, initial_profiles=d.initial_profiles, label=label,
                                     speed=self.speed, steady_state_check=steady_state_check)
        return kinetic_slice_result(time, fr)