=================
CyIpopt Changelog
=================

:Info: Change log for CyIpopt releases.
:Date: 2024-08-14
:Version: 1.6.1

GitHub holds releases, too
--------------------------

More information can be found on GitHub in the `releases section
<https://github.com/mechmotum/cyipopt/releases>`_.

About this Changelog
--------------------

All notable changes to this project will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/). Dates
should be (year-month-day) to conform with [ISO
8601](https://www.iso.org/iso-8601-date-and-time-format.html).

Formatting a New Version
------------------------

Include sections:

- Added - for new features.
- Changed - for changes in existing functionality.
- Deprecated - for soon-to-be removed features.
- Removed - for now removed features.
- Fixed - for any bug fixes.
- Security - in case of vulnerabilities.

Version History
---------------

[1.6.1] - 2025-08-14
~~~~~~~~~~~~~~~~~~~~

Removed
+++++++

- Removed unit test for ipopt package import.

[1.6.0] - 2025-08-14
~~~~~~~~~~~~~~~~~~~~

Changed
+++++++

- Bumped minimal dependency versions to match versions in Ubuntu 24.04 LTS.
  #294
- Installing and linking to the Conda Forge Ipopt binary on Windows now
  requires pkg-config (same as Linux and Mac). It is no longer required to set
  ``IPOPTWINDIR=USECONDAFORGEIPOPT``; the flag is ignored. #293
- When linking to Ipopt from a user specified directory on Windows, all dlls
  are collected instead of those with specific names making the installation
  robust to dll name changes in Ipopt versions. #288

Removed
+++++++

- All cyipopt 1.1.0 deprecated features are now removed, including the ipopt
  package. #275

Fixed
+++++

- Bugs in warm start input checking fixed and documentation added. #271
- Documentation on callbacks improved. #274

[1.5.0] - 2024-09-08
~~~~~~~~~~~~~~~~~~~~

Added
+++++

- Added instructions for using the HSL solvers on Windows. #254

Changed
+++++++

- Dropped support for Python 3.8. #263
- Dropped support for building the package with NumPy < 1.25. #263

[1.4.1] - 2024-04-02
~~~~~~~~~~~~~~~~~~~~

Fixed
+++++

- Addressed regression in return value of ``intermediate_cb``. #250

[1.4.0] - 2024-04-01
~~~~~~~~~~~~~~~~~~~~

Added
+++++

- Support for building with Cython 3. #227, #240
- Exposed the ``eps`` kwarg in the SciPy interface. #228
- Added the examples to the source tarball. #242
- Documentation improvements on specifics of Jacobian and Hessian inputs.  #247
- Support for Python 3.12.

Fixed
+++++

- Ensure ``tol`` is always a float in the SciPy interface. #236
- ``print_level`` allows integers other than 0 or 1. #244

[1.3.0] - 2023-09-23
~~~~~~~~~~~~~~~~~~~~

Added
+++++

- Added a ``pyproject.toml`` file with build dependencies. #162
- Added support for sparse Jacobians in the SciPy interface. #170
- Added ``get_current_iterate`` and ``get_current_violations`` methods to
  Problem class. #182
- Added installation instructions for Ubuntu 22.04 LTS apt dependencies.
- Added a script to build manylinux wheels. #189
- Improved documentation of ``minimize_ipopt()``. #194
- Added support for all SciPy ``minimize()`` methods. #200
- Added support for SciPy style bounds in ``minimize_ipopt()`` and added input
  validation. #207
- Added new ``CyIpoptEvaluationError`` and included it in relevance callbacks.
  #215
- Added dimension checks for Jacobian and Hessian attributes/methods. #216

Fixed
+++++

- Fixed import of ``MemoizeJac`` from scipy.optimize. #183
- ``args`` and ``kwargs`` can be passed to all functions used in
  ``minimize_ipopt()``. #197
- Fixed late binding bug in ``minimize_ipopt()`` when defining constraint
  Jacobians. #208
- Pinned build dependency Cython to < 3. #212 #214 #223
- Fixed installation on Windows for official Ipopt binaries adjacent to
  ``setup.py``. #220

Changed
+++++++

- Changed the license to Eclipse Public 2.0. #185
- Updated all dependency pins to match those in Ubuntu 22.04 LTS. #223

[1.2.0] - 2022-11-28
~~~~~~~~~~~~~~~~~~~~

Added
+++++

- Added instructions for using the HSL binaries with the Conda Forge binaries.
- Support for Python 3.10 and 3.11.

Fixed
+++++

- Improved the type information in the JAX example.
- SciPy MemoizeJac deprecation warning handled.
- Handled KeyErrors upon unknown IPOPT return statuses.
- Removed unnecessary shebangs.
- Improved the Github Actions CI.

Removed
+++++++

- Dropped support for Python 3.6.

[1.1.0] - 2021-09-07
~~~~~~~~~~~~~~~~~~~~

Added
+++++

- Added support for objective and constraint Hessians and ``jac=True`` option
  for constraints in the scipy interface.
- Example added showing how to use JAX for calculating derivatives.

Changed
+++++++

- Releases have been moved to the PyPi cyipopt distribution namespace:
  https://pypi.org/project/cyipopt/. Users should now install with ``pip
  install cyipopt``. Be sure to uninstall the ``ipopt`` distribution first.

Removed
+++++++

- The six and future dependencies are removed.

[1.0.3] - 2021-04-07
~~~~~~~~~~~~~~~~~~~~

Changed
+++++++

- Changed PyPi distribution name back to ``ipopt``, as ``cyipopt`` is currently
  unavailable.

[1.0.2] - 2021-04-06
~~~~~~~~~~~~~~~~~~~~

Changed
+++++++

- Corrected the CHANGELOG.

[1.0.1] - 2021-04-06
~~~~~~~~~~~~~~~~~~~~

Changed
+++++++

- Corrected the PyPi classifier.

[1.0.0] - 2021-04-06
~~~~~~~~~~~~~~~~~~~~

Added
+++++

- ``conda/cyipopt-dev.yml`` conda environment file for development.
- ``minimize_ipopt`` approximates the Jacobian of the objective and the
  constraints using SciPy's ``approx_fprime`` if not provided [`#91`_].
- Make changes as outlined in Version 1.0 proposal [`#14`_].
- ``requirements.txt`` file.
- Dedicated tests using pytest in ``cyipopt/tests/`` directory.
- ``examples/`` directory.
- Support for Python 3.9.
- Minimum version requirements for all dependencies.

.. _#91: https://github.com/mechmotum/cyipopt/issues/91
.. _#14: https://github.com/mechmotum/cyipopt/issues/14

Changed
+++++++

- Installation and development documentation moved from ``README.rst`` to
  ``docs/``.
- Python logger changed to use the ``cyipopt`` namespace [`#102`_].
- Class and method names now use PEP8 standards. Old class and method names now
  result in a deprecation warning.
- Module directory renamed from ``ipopt.`` to ``cyipopt``.
- ``doc/`` folder renamed to ``docs/``.
- Updated ``CHANGELOG.rst``.

.. _#102: https://github.com/mechmotum/cyipopt/issues/102

Deprecated
++++++++++

- Package being imported by ``import ipopt`` (replaced by ``import cyipopt``).
- Use of non-PEP8 named classes/function/methods, e.g. ``cyipopt.problem``
  (replaced by ``cyipopt.Problem``), ``cyipopt.problem.addOption`` (replaced by
  ``cyipopt.Problem.add_option``), ``cyipopt.problem.setProblemScaling``
  (replaced by ``cyipopt.Problem.set_problem_scaling``) etc.

Removed
+++++++

- ``test/`` folder containing examples, which have mostly been moved to
  ``examples/``
- ``docker/``, ``vagrant/`` and ``Makefile`` [`#83`_].
- Support for Python 2.7.
- Support for Python 3.5.

.. _#83: https://github.com/mechmotum/cyipopt/issues/83

[0.3.0] - 2020-12-01
~~~~~~~~~~~~~~~~~~~~

- Added support for Ipopt >=3.13 on Windows [PR `#63`_].
- Added support for Conda Forge Windows Ipopt >=3.13 binaries using the
  ``IPOPTWINDIR="USECONDAFORGEIPOPT"`` environment variable value [PR `#78`_].

.. _#63: https://github.com/mechmotum/cyipopt/pull/63
.. _#78: https://github.com/mechmotum/cyipopt/pull/78

[0.2.0] - 2020-06-05
~~~~~~~~~~~~~~~~~~~~

- Resolved compatibility issues with Windows [PR `#49`_].
- Adding installation testing on the Appveyor CI service [PR `#50`_].
- Drop Python 3.4 support and add Python 3.7 support [PR `#51`_].
- Improvements to the README and setup.py for Windows installations [PR `#54`_].
- OSError now raised if pkg-config can't find Ipopt on installation [PR `#57`_].
- Supporting only Python 2.7 and 3.6-3.8. Python 3.5 support dropped [PR `#58`_].
- Added custom installation instructions for Ubuntu 18.04.

.. _#49: https://github.com/mechmotum/cyipopt/pull/49
.. _#50: https://github.com/mechmotum/cyipopt/pull/50
.. _#51: https://github.com/mechmotum/cyipopt/pull/51
.. _#54: https://github.com/mechmotum/cyipopt/pull/54
.. _#57: https://github.com/mechmotum/cyipopt/pull/57
.. _#58: https://github.com/mechmotum/cyipopt/pull/58

[0.1.9] - 2019-09-24
~~~~~~~~~~~~~~~~~~~~

- Fixed encoding issue preventing installation on some OSes.
- Removed SciPy requirements from examples.

[0.1.8] - 2019-09-22
~~~~~~~~~~~~~~~~~~~~

- Updated ``setup.py`` to be complete and added dependencies.
- Added support for Travis CI to test build, install, examples, and docs.
- Made SciPy and optional dependency.
- Linux/Mac installation now supported via conda and conda-forge.
- Added ``LICENSE`` file and EPL headers to each source file.
- Fixed some Python 2/3 compatibility issues.
- Improved documentation formatting for Sphinx.
- Strings can be passed to addOption instead of bytes strings for Python 2 and
  3.
