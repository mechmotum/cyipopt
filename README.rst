==================
README for cyipopt
==================

Ipopt_ (Interior Point OPTimizer, pronounced eye-pea-opt) is a software package
for large-scale nonlinear optimization. Ipopt is available from the COIN-OR_
initiative, under the Eclipse Public License (EPL).

**cyipopt** is a Python wrapper around Ipopt. It enables using Ipopt from the
comfort of the Python programming language.

.. _Ipopt: https://projects.coin-or.org/Ipopt
.. _COIN-OR: https://projects.coin-or.org/

Status
======

.. list-table::

   * - Anaconda
     - .. image:: https://anaconda.org/conda-forge/cyipopt/badges/version.svg
          :target: https://anaconda.org/conda-forge/cyipopt
       .. image:: https://anaconda.org/conda-forge/cyipopt/badges/downloads.svg
          :target: https://anaconda.org/conda-forge/cyipopt
   * - PyPI
     - .. image:: https://badge.fury.io/py/ipopt.svg
          :target: https://pypi.org/project/ipopt
       .. image:: https://pepy.tech/badge/ipopt
          :target: https://pypi.org/project/ipopt
   * - Read the Docs
     - .. image:: https://readthedocs.org/projects/cyipopt/badge/?version=latest
          :target: https://cyipopt.readthedocs.io/en/latest/?badge=latest
          :alt: Documentation Status
   * - Travis CI
     - .. image:: https://api.travis-ci.org/mechmotum/cyipopt.svg?branch=master
          :target: https://travis-ci.org/mechmotum/cyipopt
   * - Appveyor
     - .. image:: https://ci.appveyor.com/api/projects/status/0o5yuogn3jx157ee?svg=true
          :target: https://ci.appveyor.com/project/moorepants/cyipopt

History
=======

**This repository was forked from https://bitbucket.org/amitibo/cyipopt and is
now considered the primary repository.** The fork includes a SciPy-style
interface and ability to handle exceptions in the callback functions.

Installation
============

We recommend using conda to install cyipopt on Linux, Mac, and Windows::

   conda install -c conda-forge cyipopt

Other `installation options`_ are present in the documentation.

.. _installation options: https://github.com/mechmotum/cyipopt/blob/master/docs/source/install.rst

License
=======

cyipopt is open-source code released under the EPL_ license, see the
``LICENSE`` file.

.. _EPL: http://www.eclipse.org/legal/epl-v10.html

Contributing
============

For bug reports, feature requests, comments, patches use the GitHub issue
tracker and/or pull request system.
