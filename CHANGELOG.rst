=================
CyIpopt Changelog
=================

:Info: Change log for CyIpopt releases.
:Date: 2021-04-06
:Version: 1.0.2

GitHub holds releases, too
--------------------------

More information can be found on GitHub in the `releases section
<https://github.com/mechmotum/cyipopt/releases>`_.

About this Changelog
--------------------

All notable changes to this project will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Dates should be (year-month-day) to conform with [ISO
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
