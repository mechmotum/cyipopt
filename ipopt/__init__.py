# -*- coding: utf-8 -*-
"""
cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

| Copyright (C) 2012-2015 Amit Aides
| Copyright (C) 2015-2018 Matthias Kümmerer

| Author: Matthias Kümmerer <matthias.kuemmerer@bethgelab.org>
| (original Author: Amit Aides <amitibo@tx.technion.ac.il>)
| URL: https://github.com/matthias-k/cyipopt
| License: EPL 1.0

Ipopt (Interior Point Optimizer, pronounced ''Eye-Pea-Opt'') is an open source
software package for large-scale nonlinear optimization. It is designed to find
(local) solutions of mathematical optimization problems of the form

.. math::

       \min_ {x \in R^n} f(x)

subject to

.. math::

       g_L \leq g(x) \leq g_U

       x_L \leq  x  \leq x_U

Where :math:`x` are the optimization variables (possibly with upper an lower
bounds), :math:`f(x)` is the objective function and :math:`g(x)` are the
general nonlinear constraints. The constraints, :math:`g(x)`, have lower and
upper bounds. Note that equality constraints can be specified by setting
:math:`g^i_L = g^i_U`.

**cyipopt** is a python wrapper around Ipopt. It enables using Ipopt from the
comfort of the Python programming language. cyipopt is available under the EPL
(Eclipse Public License) open-source license.

"""

from __future__ import absolute_import

from cyipopt import *
from .ipopt_wrapper import minimize_ipopt
from .version import __version__
