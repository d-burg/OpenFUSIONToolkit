#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
# TokaMaker tests: Fortran namelist reader behind read_kfile and read_mhdin (pure Python, synthetic files)
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python')))
from OpenFUSIONToolkit.util import read_fortran_namelist


def write_text(tmp_path, text, name='input.nml'):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_namelist_values_are_literals(tmp_path):
    '''Single values are read as literals: integers, reals and quoted strings keep their types, and a token that
    is not a literal (an expression or a name defined in the reader's module) comes back as its text'''
    path = write_text(tmp_path, "&IN1\n NI = 3\n XR = 2.5\n XE = 1.0E-3\n NAME = 'DEV-1'\n LOG = T\n"
                                " EXPR = abs(-2)\n CONST = mu0\n LAST = 0\n/\n")
    nml = read_fortran_namelist(path)
    assert nml['NI'] == 3 and isinstance(nml['NI'], int)
    assert nml['XR'] == 2.5 and nml['XE'] == 1.0E-3
    assert nml['NAME'] == 'DEV-1'
    assert nml['LOG'] == 'T'
    assert nml['EXPR'] == 'abs(-2)'
    assert nml['CONST'] == 'mu0'


def test_namelist_several_assignments_per_line(tmp_path):
    '''Several assignments on one line, including after an array that continues from the previous line, are
    read as separate names'''
    path = write_text(tmp_path, "&IN1\n NA=1 XB=2.5, NAME='x'\n ARR = 1.0, 2.0,\n   3.0 NE=3\n LAST = 0\n/\n")
    nml = read_fortran_namelist(path)
    assert nml['NA'] == 1 and nml['XB'] == 2.5 and nml['NAME'] == 'x'
    assert np.array_equal(nml['ARR'], [1.0, 2.0, 3.0])
    assert nml['NE'] == 3


def test_namelist_repeated_names(tmp_path):
    '''A name that appears again in a later group keeps the value of its first group, and the names read after
    it keep their own values; a name assigned twice within one group takes the later value (Fortran)'''
    path = write_text(tmp_path, "&IN1\n WT = 1.0, 0.0, 1.0\n NX = 5\n/\n&EXTRA\n WT = 3*1.0\n NY = 7\n"
                                " NZ = 8\n NV = 1\n NV = 2\n LAST = 0\n/\n")
    nml = read_fortran_namelist(path)
    assert np.array_equal(nml['WT'], [1.0, 0.0, 1.0])
    assert nml['NX'] == 5 and nml['NY'] == 7 and nml['NZ'] == 8
    assert nml['NV'] == 2


def test_namelist_strings_and_group_delimiters(tmp_path):
    '''Values on a group's header or terminator line are read; quoted strings may hold slashes, commas and "!";
    text between groups is ignored; groups may also end with &END or $END'''
    path = write_text(tmp_path, "&IN1 NI=4\n PATH = '/a/b/'\n NAME = 'x!y,z' ! comment\n XA = 1.0 /\n"
                                "free text between groups\n&IN2\n XB = 2.0\n&END\n$IN3\n NC = 3\n LAST = 0\n$END\n")
    nml = read_fortran_namelist(path)
    assert nml['NI'] == 4
    assert nml['PATH'] == '/a/b/' and nml['NAME'] == 'x!y,z'
    assert nml['XA'] == 1.0 and nml['XB'] == 2.0 and nml['NC'] == 3


MHDIN_SEVERAL_PER_LINE = """&IN3
 MPNAM2 = 'PRB1', 'PRB2', LPNAME = 'LOOP1', 'LOOP2' AMP2 = 0.0, 90.0
 FCID = 1., 2.  ECID = 1., 1.
 RF = 1.0, 1.1, ZF = 0.5, -0.5, WF = 0.1, 0.1, HF = 0.2, 0.2, TURNFC = 10., 20.
 RE = 0.5, 0.5 ZE = 0.1, -0.1 WE = 0.05, 0.05 HE = 0.3, 0.3
/
"""


def test_read_mhdin_several_assignments_per_line(tmp_path):
    '''read_mhdin on a machine file that puts several assignments on a line returns the names, coil geometry
    and probe angles written in it'''
    from OpenFUSIONToolkit.TokaMaker.util import read_mhdin
    machine, _ = read_mhdin(write_text(tmp_path, MHDIN_SEVERAL_PER_LINE, 'mhdin.dat'))
    assert machine['MPNAM2'] == ['PRB1', 'PRB2'] and machine['LPNAME'] == ['LOOP1', 'LOOP2']
    assert list(machine['AMP2'].values()) == [0.0, 90.0]
    assert machine['FCOIL']['FCOIL001'] == [1.0, 0.5, 0.1, 0.2, 10.0]
    assert machine['FCOIL']['FCOIL002'] == [1.1, -0.5, 0.1, 0.2, 20.0]
    assert machine['ECOIL']['ECOIL001'] == [[0.5, 0.1, 0.05, 0.3], [0.5, -0.1, 0.05, 0.3]]


KFILE_TWO_GROUPS = """&IN1
 COILS = 1.0, 2.0
 FWTSI = 1.0, 0.0
 EXPMP2 = 0.1, 0.2  FWTMP2 = 1.0, 0.0
 BRSP = 10., 20.
 FWTFC = 0., 1.
 ECURRT = 5., 6.
 FWTEC = 1., 1.
/
&EXTRA
 FWTSI = 2*1.0   COILS = 9.0 9.0
 FWTMP2 = 2*1.0
/
"""


def test_read_kfile_trailing_group(tmp_path):
    '''read_kfile on a k-file whose trailing group repeats measurement weights (several assignments per line)
    returns the values and weights of the first group'''
    from OpenFUSIONToolkit.TokaMaker.util import read_kfile
    machine = {'MPNAM2': ['PRB1', 'PRB2'], 'LPNAME': ['LOOP1', 'LOOP2'],
               'ECOIL': {'ECOIL001': [], 'ECOIL002': []}, 'FCOIL': {'FCOIL001': [], 'FCOIL002': []}}
    probes, loops, ecoils, fcoils, _ = read_kfile(write_text(tmp_path, KFILE_TWO_GROUPS, 'k.in'), machine)
    assert [list(v) for v in probes.values()] == [[0.1, 1.0], [0.2, 0.0]]
    assert [list(v) for v in loops.values()] == [[1.0, 1.0], [2.0, 0.0]]
    assert [list(v) for v in fcoils.values()] == [[10.0, 0.0], [20.0, 1.0]]
    assert [list(v) for v in ecoils.values()] == [[5.0, 1.0], [6.0, 1.0]]


def test_namelist_double_precision_exponents(tmp_path):
    '''Fortran double precision exponents (D or d) are read as numbers, alone, in arrays and in repeats'''
    path = write_text(tmp_path, "&IN1\n XS = 1.5D-03\n XA = 1.0D0, 2.0d+1\n XR = 2*1.0D2\n LAST = 0\n/\n")
    nml = read_fortran_namelist(path)
    assert nml['XS'] == 1.5E-3
    assert np.array_equal(nml['XA'], [1.0, 20.0]) and np.array_equal(nml['XR'], [100.0, 100.0])


def test_namelist_null_values(tmp_path):
    '''A null value (an empty entry between separators, or a repeat count with no value), which Fortran leaves
    unchanged, keeps its position in the array as NaN; a trailing separator is not a null'''
    path = write_text(tmp_path, "&IN1\n XN = 1.0, , 3.0\n XM = 2*, 5.0\n XT = 1.0, 2.0,\n LAST = 0\n/\n")
    nml = read_fortran_namelist(path)
    assert np.array_equal(nml['XN'], [1.0, np.nan, 3.0], equal_nan=True)
    assert np.array_equal(nml['XM'], [np.nan, np.nan, 5.0], equal_nan=True)
    assert np.array_equal(nml['XT'], [1.0, 2.0])


def test_namelist_unchanged_conventions(tmp_path):
    '''Conventions the reader keeps: names keep their case, repeat counts and continued lines give arrays,
    a trailing comma adds nothing, an empty value is an empty string, logicals stay text, string arrays are
    returned as their text, and the last assignment of the file is returned as an array'''
    path = write_text(tmp_path, "&IN1\n abc = 1\n ABC = 2\n REP = 3*0.5 ! comment\n CONT = 1.0,\n   2.0,\n"
                                " EMPTY =\n LOGS = .T., F\n NAMES = 'A1', 'B2'\n LAST = 7\n/\n")
    nml = read_fortran_namelist(path)
    assert nml['abc'] == 1 and nml['ABC'] == 2
    assert np.array_equal(nml['REP'], [0.5, 0.5, 0.5]) and np.array_equal(nml['CONT'], [1.0, 2.0])
    assert nml['EMPTY'] == ''
    assert nml['LOGS'].split() == ['.T.', 'F']
    assert nml['NAMES'].replace("'", ' ').split() == ['A1', 'B2']
    assert isinstance(nml['LAST'], np.ndarray) and np.array_equal(nml['LAST'], [7.0])


def test_namelist_bottom_table(tmp_path):
    '''With b_arr=True the table after the namelist gives the F-coil and vessel geometry (older machine files)'''
    path = write_text(tmp_path, "&IN3\n TURNFC = 1., 1.\n RSISVS = 1*0.\n IECOIL = 0\n/\n"
                                " 1.0 0.5 0.1 0.2 0.0 0.0\n 1.1 -0.5 0.1 0.2 0.0 0.0\n 2.0 0.0 0.05 0.05 0.0 0.0\n")
    nml = read_fortran_namelist(path, b_arr=True)
    assert nml['b_arr'].shape == (3, 6)
    assert np.array_equal(nml['RF'], [1.0, 1.1]) and np.array_equal(nml['ZF'], [0.5, -0.5])
    assert np.array_equal(nml['RVS'], [2.0]) and np.array_equal(nml['HVS'], [0.05])
