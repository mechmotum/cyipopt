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

All of the dependencies can be installed with Ubuntu's package manager::

   sudo apt install build-essential python-dev python-six cython python-numpy coinor-libipopt1v5 coinor-libipopt-dev

Note that six, cython, and numpy could alternatively be installed using Python
specific package managers, e.g. ``pip install six cython numpy``.

The NumPy and IPOPT libs and headers are installed in standard locations, so
you should not need to set ``LD_LIBRARY_PATH`` or ``PKG_CONFIG_PATH``.

Now run ``python setup.py build`` to compile cyipopt. In the output of this
command you should see two calls to ``gcc`` for compiling and linking. Make
sure both of these are pointing to the correct libraries and headers. They will
look something like this (formatted and commented for easy viewing here)::

   $ python setup.py build
   ...
   x86_64-linux-gnu-gcc -pthread -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -fno-strict-aliasing
     -Wdate-time -D_FORTIFY_SOURCE=2 -g -fdebug-prefix-map=/build/python2.7-3hk45v/python2.7-2.7.15~rc1=.
     -fstack-protector-strong -Wformat -Werror=format-security -fPIC
     -I/usr/local/include/coin  # points to IPOPT headers
     -I/usr/local/include/coin/ThirdParty  # points to IPOPT third party headers
     -I/usr/lib/python2.7/dist-packages/numpy/core/include  # points to NumPy headers
     -I/usr/include/python2.7  # points to Python 2.7 headers
     -c src/cyipopt.c -o build/temp.linux-x86_64-2.7/src/cyipopt.o
   x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro
     -fno-strict-aliasing -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -Wdate-time -D_FORTIFY_SOURCE=2 -g
     -fdebug-prefix-map=/build/python2.7-3hk45v/python2.7-2.7.15~rc1=. -fstack-protector-strong -Wformat
     -Werror=format-security -Wl,-Bsymbolic-functions -Wl,-z,relro -Wdate-time -D_FORTIFY_SOURCE=2 -g
     -fdebug-prefix-map=/build/python2.7-3hk45v/python2.7-2.7.15~rc1=. -fstack-protector-strong -Wformat
     -Werror=format-security build/temp.linux-x86_64-2.7/src/cyipopt.o
     -L/usr/local/lib
     -L/lib/../lib
     -L/usr/lib/../lib
     -L/usr/lib/gcc/x86_64-linux-gnu/5
     -L/usr/lib/gcc/x86_64-linux-gnu/5/../../..
     -L/usr/lib/gcc/x86_64-linux-gnu/5/../../../../lib
     -L/usr/lib/gcc/x86_64-linux-gnu/5/../../../x86_64-linux-gnu
     -lipopt -llapack -lblas -lm -ldl -lcoinmumps -lblas -lgfortran -lm -lquadmath  # linking to relevant libs
     -lcoinhsl -llapack -lblas -lgfortran -lm -lquadmath -lcoinmetis  # linking to relevant libs
     -o build/lib.linux-x86_64-2.7/cyipopt.so
   ...

You can check that everything linked correctly with ``ldd``::

   $ ldd build/lib.linux-x86_64-2.7/cyipopt.so
   linux-vdso.so.1 (0x00007ffc1677c000)
   libipopt.so.0 => /usr/local/lib/libipopt.so.0 (0x00007fcdc8668000)
   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fcdc8277000)
   libcoinmumps.so.0 => /usr/local/lib/libcoinmumps.so.0 (0x00007fcdc7eef000)
   libcoinhsl.so.0 => /usr/local/lib/libcoinhsl.so.0 (0x00007fcdc7bb4000)
   liblapack.so.3 => /usr/lib/x86_64-linux-gnu/liblapack.so.3 (0x00007fcdc732e000)
   libblas.so.3 => /usr/lib/x86_64-linux-gnu/libblas.so.3 (0x00007fcdc70d3000)
   libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fcdc6ecf000)
   libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fcdc6b46000)
   libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fcdc67a8000)
   /lib64/ld-linux-x86-64.so.2 (0x00007fcdc8d20000)
   libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fcdc6590000)
   libcoinmetis.so.0 => /usr/local/lib/libcoinmetis.so.0 (0x00007fcdc6340000)
   libgfortran.so.3 => /usr/lib/x86_64-linux-gnu/libgfortran.so.3 (0x00007fcdc600f000)
   libopenblas.so.0 => /usr/lib/x86_64-linux-gnu/libopenblas.so.0 (0x00007fcdc3d69000)
   libgfortran.so.4 => /usr/lib/x86_64-linux-gnu/libgfortran.so.4 (0x00007fcdc398a000)
   libquadmath.so.0 => /usr/lib/x86_64-linux-gnu/libquadmath.so.0 (0x00007fcdc374a000)
   libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fcdc352b000)

And finally::

   $ python setup.py install

Note that you may or may not want to install this package system wide. It is
best to install into your user space, i.e. what ``pip install --user`` does, or
setup a virtual environment with tools like venv or conda. If you use virtual
environments you will need to be careful about selecting headers and libraries
for packages in or out of the virtual environments in the build step.

Docker container
================

The subdirectory ``docker`` contains a docker container with preinstalled ipopt
and cyipopt.  To build the container, cd into the ``docker`` directory and run
``make``. Then you can start the container by::

   $ docker run -it matthiask/ipopt /bin/bash

and either call ``ipopt`` directly or start a ipython shell and ``import ipopt``.


Manually compile IPOPT and Cyipopt (On Ubuntu 18.04.1 LTS, python 2.7.15)
-------------------------------------

Before start, make sure ``pip`` is installed. If not, please install it.

    ``sudo apt install python-pip``

Then use ``pip`` install the following packages:

    $ pip2 install pkgconfig numpy scipy cython six future wget
    
    
* Compile Ipopt

1. Download ipopt source code, choose the verison that you would like to have from this page <https://www.coin-or.org/download/source/Ipopt/>.

Here I used ``Ipopt-3.12.11``

Extract ipopt and put it in the floder you want (I put it at ``Home`` folder).

Create the Ipopt directory: ``export IPOPTDIR=/home/huawei/Ipopt-3.12.11``

2. Download external packages using ``wget``::

    cd $IPOPTDIR/ThirdParty/Blas
    ./get.Blas
    cd ../Lapack && ./get.Lapack
    cd ../ASL && ./get.ASL
    cd ../Mumps && ./get.Mumps
    cd ../Metis && ./get.Metis

To use ``ma27, ma57, ma86`` solvers, ``HSL`` package are needed. ``HSL`` can be get from its official website <http://www.hsl.rl.ac.uk/ipopt/>, academic license is free.

Extract ''HSL'' source code after you get it. Rename the extracted folder to ``coinhsl`` and copy it in the HSL folder: ``Ipopt-3.12.11/ThirdParty/HSL``

3. Install Ipopt using ``./configure, make, make install`` commands::
    
    $ mkdir $IPOPTDIR/build
    $ cd $IPOPTDIR/build 
    $ IPOPTDIR/configure
    $ make 
    $ make test 
    $ make install

I used the ipopt document as reference, and it descrips the compile process in a very detail way. 
If you get errors in this process, please check it <https://www.coin-or.org/Ipopt/documentation/node14.html>

4. Set environment path::

    $ export IPOPTPATH="/home/huawei/Ipopt-3.12.11/build" # make sure this one starts with the Home path
    $ export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$IPOPTPATH/lib/pkgconfig 
    $ export PATH=$PATH:$IPOPTPATH/bin

Get help from this web-page if you get errors in setting environments. <https://stackoverflow.com/questions/13428910/how-to-set-the-environmental-variable-ld-library-path-in-linux>


* Compile cyipopt

1. Download ``cyipopt`` source code from this github: <https://github.com/matthias-k/cyipopt>

Extract cyipopt and put it at the place you want (I put it at Home directory).

Change the path of terminal to cyipopt: ``cd ~/cyipopt``

Compile cyipopt using the command: ``python setup.py install``

If there is no error, then you have compiled ``cyipopt`` successfully

2. check that everything linked correctly with ``ldd`` ::

    $ ldd build/lib.linux-x86_64-2.7/cyipopt.so
    linux-vdso.so.1 (0x00007ffe895e1000)
    libipopt.so.1 => /home/huawei/Ipopt-3.12.11/build/lib/libipopt.so.1 (0x00007f74efc2a000)
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f74ef839000)
    libcoinmumps.so.1 => /home/huawei/Ipopt-3.12.11/build/lib/libcoinmumps.so.1 (0x00007f74ef4ae000)
    libcoinhsl.so.1 => /home/huawei/Ipopt-3.12.11/build/lib/libcoinhsl.so.1 (0x00007f74ef169000)
    liblapack.so.3 => /usr/lib/x86_64-linux-gnu/liblapack.so.3 (0x00007f74ee8cb000)
    libblas.so.3 => /usr/lib/x86_64-linux-gnu/libblas.so.3 (0x00007f74ee65e000)
    libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f74ee45a000)
    libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f74ee0d1000)
    libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f74edd33000)
    /lib64/ld-linux-x86-64.so.2 (0x00007f74f02c0000)
    libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f74edb1b000)
    libcoinmetis.so.1 => /home/huawei/Ipopt-3.12.11/build/lib/libcoinmetis.so.1 (0x00007f74ed8ca000)
    libgfortran.so.4 => /usr/lib/x86_64-linux-gnu/libgfortran.so.4 (0x00007f74ed4eb000)
    

3. Before try the test code, add Ipopt ``lib`` path to ``LD_LIBRARY_PATH``::

    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/huawei/Ipopt-3.12.11/build/lib

To make this path works for all terminal, it can be added to ``.bashrc`` ::

    $ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/huawei/Ipopt-3.12.11/build/lib' >> ~/.bashrc

4. Now you can run the test code::

    $ cd test
    $ python -c "import ipopt"
    $ python examplehs071.py

If it could be run successfully, the optimization will start with the following descriptions::

    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit http://projects.coin-or.org/Ipopt
    ******************************************************************************

    This is Ipopt version 3.12.11, running with linear solver ma27.

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
