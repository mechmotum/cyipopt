Welcome to cyipopt's documentation!
===================================

cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

Ipopt_ (Interior Point Optimizer, pronounced ''Eye-Pea-Opt'') is an open source
software package for large-scale nonlinear optimization. It is designed to find
(local) solutions of mathematical optimization problems of the form

.. math::

       \min_{x \in R^n} f(x)

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

.. _Ipopt: https://coin-or.github.io/Ipopt/

Contents:

.. toctree::
   :maxdepth: 2

   install
   tutorial
   reference
   development

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Copyright
=========

| Copyright (C) 2012-2015 Amit Aides
| Copyright (C) 2015-2017 Matthias Kümmerer
| Copyright (C) 2017-2021 cyipopt developers
| License: EPL 1.0
