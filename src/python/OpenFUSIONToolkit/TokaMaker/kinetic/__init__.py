#------------------------------------------------------------------------------
# Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
#
# SPDX-License-Identifier: LGPL-3.0-only
#------------------------------------------------------------------------------
'''! Modules of the single-slice kinetic equilibrium reconstruction with TokaMaker

The modules take arrays and device data from the caller; nothing device-specific is defined here.
The kinetic-input, coordinate, pressure, edge and current-model helpers import only numpy, scipy
and the standard library.  The forward model (``forward``), the solver-evaluated bootstrap
(``bootstrap``) and measurement rows (``solver_rows``), the scoring (``score``) and the fit (``fit``)
drive a ``TokaMaker`` object.

@authors Daniel Burgess
@date November 2026
@ingroup doxy_oft_python
'''
