This repository has been forked from <https://bitbucket.org/amitibo/cyipopt>.
The code is now able to handle exceptions in the callback functions. Also,
a docker container for easy usage is provided.

==================
README for cyipopt
==================

Ipopt (Interior Point OPTimizer, pronounced eye-pea-Opt) is a software package
for large-scale nonlinear optimization.

cyipopt is a python wrapper around Ipopt. It enables using Ipopt from the
comfort of the great Python scripting language.

Ipopt is available from the `COIN-OR <https://projects.coin-or.org/Ipopt>`_
initiative, under the Eclipse Public License (EPL).


Usage
=====

For simple cases where you do not need the full power of sparse and structured jacobians etc,
`cyipopt` provides the function `minimize_ipopt` which has the same behaviour as
`scipy.optimize.minimize`

::

    from scipy.optimize import rosen, rosen_der
    from ipopt import minimize_ipopt
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    res = minimize_ipopt(rosen, x0, jac=rosen_der)
    print(res)



Installing
==========

To install cyipopt you will need the following prerequisites:

  * python 2.6+
  * numpy
  * scipy
  * cython
  * C/C++ compiler

The [Anaconda Python Distribution](https://www.continuum.io/why-anaconda) is
one of the easiest ways to install python for Linux, Mac and Windows.

You will also need the binaries and header files of the Ipopt package. I
recommend downloading the `binaries <http://www.coin-or.org/download/binary/Ipopt/>`_
especially as they include a version compiled against the MKL library.

Download the source files of cyipopt and update ``setup.py`` to point to the header
files and binaries of the Ipopt package, if `LD_LIBRARY_PATH` and `pkg_config` are
not setup to find ipopt on their own.

Then, execute::

   python setup.py install

Docker container
================

The subdirectory `docker` contains a docker container with preinstalled ipopt and cyipopt.
To build the container, cd into the `docker` directory and run `make`. Then you can
start the container by::

   docker run -it matthiask/ipopt /bin/bash

and either call `ipopt` directly or start a ipython shell and `import ipopt`.

Vagrant environment
===================

The subdirectory `vagrant` contains a `Vagrantfile` that installs ipopt and cyipopt in OS provision.
To build the environment, cd into the `vagrant` directory and run `vagrant up` (Requires that you have Vagrant+VirtualBox installed). Then you can
access the system by::

   vagrant ssh

and either call `ipopt` directly or start a python shell and `import ipopt`.
Also, if you get `source files <http://www.coin-or.org/download/binary/Ipopt/>` of coinhsl and put it
in the `vagrant` directory, the vagrant provision will detect and add them in the ipopt compiling process, and
then you will have ma57, ma27, and other solvers available on ipopt binary (ma97 and mc68 were removed to avoid compilation errors).

Reading the docs
================

After installing::

   cd doc
   make html

Then, direct your browser to ``build/html/index.html``.


Testing
=======

You can test the installation by running the examples under the folder ``test\``.


Conditions of use
=================

cyipopt is open-source code released under the
`EPL <http://www.eclipse.org/legal/epl-v10.html>`_ license.


Contributing
============

For bug reports use the github issue tracker.
You can also send wishes, comments, patches, etc. to matthias.kuemmerer@bethgelab.org
