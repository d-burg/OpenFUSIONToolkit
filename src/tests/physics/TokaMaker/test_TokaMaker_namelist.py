#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
# TokaMaker tests: Fortran namelist reader behind read_kfile and read_mhdin (pure Python, synthetic files)
import os
import sys

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
