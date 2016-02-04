# -*- coding: utf-8 -*-
"""
ipopt - A cython wrapper for the IPOPT optimization solver.
===========================================================

IPOPT (Interior Point Optimizer, pronounced ''Eye-Pea-Opt'') is an open source
software package for large-scale nonlinear optimization. It is designed to find
(local) solutions of mathematical optimization problems of the from

.. math::

       \min_ {x \in R^n} f(x)

subject to

.. math::

       g_L \leq g(x) \leq g_U

       x_L \leq  x  \leq x_U

Where :math:`x` are the optimization variables (possibly with upper an lower
bounds), :math:`f(x)` is the objective function and :math:`g(x)` are the general nonlinear
constraints. The constraints, :math:`g(x)`, have lower and upper bounds. Note that
equality constraints can be specified by setting :math:`g^i_L = g^i_U`.

**cyipopt** is a python wrapper around Ipopt. It enables using Ipopt from the comfort of the
Python scripting language. cyipopt is available under the EPL (Eclipse Public License)
open-source license.

.. codeauthor:: Amit Aides <amitibo@tx.technion.ac.il>
"""

# Author: Amit Aides <amitibo@tx.technion.ac.il>
#
# License: EPL.

from __future__ import absolute_import

from cyipopt import *
from .ipopt_wrapper import minimize_ipopt

