**This repository was forked from https://bitbucket.org/amitibo/cyipopt and is
now considered the primary repository.** The fork includes a SciPy-style
interface, ability to handle exceptions in the callback functions, and docker
container for easy usage.

==================
README for cyipopt
==================

Ipopt_ (Interior Point OPTimizer, pronounced eye-pea-opt) is a software package
for large-scale nonlinear optimization. Ipopt is available from the COIN-OR_
initiative, under the Eclipse Public License (EPL).

cyipopt is a Python wrapper around Ipopt. It enables using Ipopt from the
comfort of the Python programming language.

.. _Ipopt: https://projects.coin-or.org/Ipopt
.. _COIN-OR: https://projects.coin-or.org/

Usage
=====

For simple cases where you do not need the full power of sparse and structured
Jacobians etc, ``cyipopt`` provides the function ``minimize_ipopt`` which has
the same behaviour as ``scipy.optimize.minimize``, for example:

.. code:: python

    from scipy.optimize import rosen, rosen_der
    from ipopt import minimize_ipopt
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    res = minimize_ipopt(rosen, x0, jac=rosen_der)
    print(res)

Installation
============

The `Anaconda Python Distribution <https://www.continuum.io/why-anaconda>`_ is
one of the easiest ways to install Python and associated packages for Linux,
Mac, and Windows. Once Anaconda (or miniconda) is installed, you can install
cyipopt on Linux and Mac from the Conda Forge channel with::

   $ conda install -c conda-forge cyipopt

The above command will install binary versions of all the necessary
dependencies and cyipopt. Note that there currently are no Windows binaries.
You will have to install from source from Windows or if you want a customized
installation, e.g. with MKL, HSL, etc.

To begin installing from source you will need to install the following
dependencies:

  * C/C++ compiler
  * pkg-config [only for Linux and Mac]
  * Ipopt
  * Python 2.7 or 3.4+
  * setuptools
  * cython
  * numpy
  * six
  * future
  * scipy [optiional]

The binaries and header files of the Ipopt package can be obtained from
http://www.coin-or.org/download/binary/Ipopt/. These include a version compiled
against the MKL library. Or you can build Ipopt from source. The remaining
dependencies can be installed with conda or other package managers.

Download the source files of cyipopt and update ``setup.py`` to point to the
header files and binaries of the Ipopt package, if ``LD_LIBRARY_PATH`` and
``pkg_config`` are not setup to find ipopt on their own.

Then, execute::

   $ python setup.py install

Installation on Ubuntu 18.04 LTS
--------------------------------

All of the dependencies can be install with Ubuntu's package manager::

   sudo apt install build-essential python-dev python-six cython python-numpy coinor-libipopt1v5 coinor-libipopt-dev

Note that six, cython, and numpy could alternatively be installed using Python
specific package managers, e.g. ``pip install six cython numpy``.

The NumPy and IPOPT libs and headers are installed in standard locations, so
you should not need to set ``LD_LIBRARY_PATH`` or ``PKG_CONFIG_PATH``.

Now run::

   $ python setup.py install

In the output of this command you should see two calls to ``gcc`` for compiling
and linking. Make sure both of these are pointing to the correct libraries and
headers. They will look something like this::

   gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -m64 -fPIC -m64 -fPIC -fPIC
     -I/usr/local/include/coin -I/usr/local/include/coin/ThirdParty  # points to IPOPT lib and headers
     -I/usr/local/lib/python2.7/site-packages/numpy/core/include  # points to NumPy headers
     -I/usr/local/include/python2.7m  # points to Python headers
     -c src/cyipopt.c -o build/temp.linux-x86_64-3.6/src/cyipopt.o
   gcc -pthread -shared
     -L/usr/local/lib -Wl,-rpath=/usr/local/lib,--no-as-needed
     -L/usr/local/lib -Wl,-rpath=/usr/local/lib,--no-as-needed build/temp.linux-x86_64-3.6/src/cyipopt.o
     -L/usr/local/lib
     -L/usr/lib/gcc/x86_64-linux-gnu/5
     -L/usr/lib/gcc/x86_64-linux-gnu/5/../../../x86_64-linux-gnu
     -L/usr/lib/gcc/x86_64-linux-gnu/5/../../../../lib
     -L/lib/../lib -L/usr/lib/../lib
     -L/usr/lib/gcc/x86_64-linux-gnu/5/../../..
     -L/usr/local/lib -L/usr/lib/gcc/x86_64-linux-gnu/5
     -L/usr/lib/gcc/x86_64-linux-gnu/5/../../../x86_64-linux-gnu
     -L/usr/lib/gcc/x86_64-linux-gnu/5/../../../../lib
     -L/lib/../lib -L/usr/lib/../lib
     -L/usr/lib/gcc/x86_64-linux-gnu/5/../../..
     -L/usr/local/lib -L/home/moorepants/miniconda3/envs/cyipopt-manual/lib
     -lipopt -llapack -lblas -lm -ldl -lcoinmumps -lblas -lgfortran -lm -lquadmath  # linking to relevant libs
     -lcoinhsl -llapack -lblas -lgfortran -lm -lquadmath -lcoinmetis -lpython3.6m  # linking to relevant libs
     -o build/lib.linux-x86_64-3.6/cyipopt.cpython-36m-x86_64-linux-gnu.so

Docker container
================

The subdirectory ``docker`` contains a docker container with preinstalled ipopt
and cyipopt.  To build the container, cd into the ``docker`` directory and run
``make``. Then you can start the container by::

   $ docker run -it matthiask/ipopt /bin/bash

and either call ``ipopt`` directly or start a ipython shell and ``import ipopt``.

Vagrant environment
===================

The subdirectory ``vagrant`` contains a ``Vagrantfile`` that installs ipopt and
cyipopt in OS provision. To build the environment, cd into the ``vagrant``
directory and run ``vagrant up`` (Requires that you have Vagrant+VirtualBox
installed). Then you can access the system by::

   $ vagrant ssh

and either call ``ipopt`` directly or start a python shell and ``import
ipopt``.  Also, if you get `source files
<http://www.coin-or.org/download/binary/Ipopt/>` of coinhsl and put it in the
``vagrant`` directory, the vagrant provision will detect and add them in the
ipopt compiling process, and then you will have ma57, ma27, and other solvers
available on ipopt binary (ma97 and mc68 were removed to avoid compilation
errors).

Reading the docs
================

After installing::

   $ cd doc
   $ make html

Then, direct your browser to ``build/html/index.html``.

Testing
=======

You can test the installation by running the examples under the folder ``test\``.

Conditions of use
=================

cyipopt is open-source code released under the EPL_ license.

.. _EPL: http://www.eclipse.org/legal/epl-v10.html

Contributing
============

For bug reports use the github issue tracker. You can also send wishes,
comments, patches, etc. to matthias.kuemmerer@bethgelab.org
