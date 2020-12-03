.. highlight:: sh

Installation
============

The `Anaconda Python Distribution <https://www.continuum.io/why-anaconda>`_ is
one of the easiest ways to install Python and associated packages for Linux,
Mac, and Windows. Once Anaconda (or conda) is installed, you can install
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
  * python 3.6+
  * setuptools
  * cython
  * numpy
  * scipy
  * six
  * future

The binaries and header files of the Ipopt package can be obtained from
http://www.coin-or.org/download/binary/Ipopt/. These include a version compiled
against the MKL library. Or you can build IPopt from source. The remaining
dependencies can be installed with conda or other package managers.

Download the source files of cyipopt and update ``setup.py`` to point to the
header files and binaries of the Ipopt package, if `LD_LIBRARY_PATH` and
`pkg_config` are not setup to find ipopt on their own.

Then, execute::

   $ python setup.py install

Docker container
================

The subdirectory `docker` contains a docker container with preinstalled ipopt
and cyipopt.  To build the container, cd into the `docker` directory and run
`make`. Then you can start the container by::

   docker run -it matthiask/ipopt /bin/bash

and either call `ipopt` directly or start a ipython shell and `import ipopt`.

Vagrant environment
===================

The subdirectory `vagrant` contains a `Vagrantfile` that installs ipopt and
cyipopt in OS provision. To build the environment, cd into the `vagrant`
directory and run `vagrant up` (Requires that you have Vagrant+VirtualBox
installed). Then you can access the system by::

   vagrant ssh

and either call `ipopt` directly or start a python shell and `import ipopt`.
Also, if you get `source files <http://www.coin-or.org/download/binary/Ipopt/>`
of coinhsl and put it in the `vagrant` directory, the vagrant provision will
detect and add them in the ipopt compiling process, and then you will have
ma57, ma27, and other solvers available on ipopt binary (ma97 and mc68 were
removed to avoid compilation errors).

Reading the docs
================

After installing::

   cd doc
   make html

Then, direct your browser to ``build/html/index.html``.

Testing
=======

You can test the installation by running the examples under the folder ``test\``.

.. note::

    Under linux you might need to let the OS know where to look for the Ipopt lib files,
    e.g. use::

        $ export LD_LIBRARY_PATH=<PATH to Ipopt lib files>

