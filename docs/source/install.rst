.. highlight:: sh

============
Installation
============

Using conda
-----------

Conda_ is a cross platform package manager and provides the easiest mechanism
to install cyipopt on Linux, Mac, and Windows. Once conda is installed, install
cyipopt from the Conda Forge channel with::

   $ conda install -c conda-forge cyipopt

The above command will install binary versions of all the necessary
dependencies as well as cyipopt. Conda Forge supplies a basic build of Ipopt
that is suitable for many use cases. You will have to install from source if
you want a customized Ipopt installation.

.. _Conda: https://docs.conda.io

From source
-----------

To begin installing from source you will need to install the following
dependencies:

  * C/C++ compiler
  * pkg-config [only for Linux and Mac]
  * Ipopt [>= 3.13 on Windows]
  * Python 3.6+
  * setuptools
  * cython
  * numpy
  * scipy [optional]

The binaries and header files of the Ipopt package can be obtained from
http://www.coin-or.org/download/binary/Ipopt/. These include a version compiled
against the MKL library. Or you can build Ipopt from source. The remaining
dependencies can be installed with conda or other package managers.

On Linux and Mac
~~~~~~~~~~~~~~~~

For Linux and Mac, the ``ipopt`` executable should be in your path and
discoverable by pkg-config, i.e. this command should return a valid result::

   $ pkg-config --libs --cflags ipopt

You will need to install Ipopt in a system location or set ``LD_LIBRARY_PATH``
if pkg-config does not find the executable.

Once all the dependencies are installed, execute::

   $ python setup.py install

to build and install the package.

From source on Windows
~~~~~~~~~~~~~~~~~~~~~~

Install the dependencies with conda (Anaconda or Miniconda)::

   $ conda.exe install -c conda-forge numpy cython setuptools

Or alternatively with pip::

   $ pip install numpy cython setuptools

Additionally, make sure you have a C compiler setup to compile Python C
extensions, e.g. Visual C++. Build tools for VS2019
https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019
have been tested to work for conda Python 3.7 (see
https://github.com/mechmotum/cyipopt/issues/52).

Download and extract the cyipopt source code from Github or PyPi.

Obtain IPOPT one of two ways:

1. Using official IPOPTs binaries:

Download the latest precompiled version of Ipopt that includes the DLL files
from https://github.com/coin-or/Ipopt/releases. Note that the current setup
only supports Ipopt >= 3.13.0. The build 3.13.3 of Ipopt has been confirmed to
work and can be downloaded from `Ipopt-3.13.3-win64-msvs2019-md.zip
<https://github.com/coin-or/Ipopt/releases/download/releases%2F3.13.3/Ipopt-3.13.3-win64-msvs2019-md.zip>`_.
After Ipopt is extracted, the ``bin``, ``lib`` and ``include`` folders should
be in the root cyipopt directory, i.e. adjacent to the ``setup.py`` file.
Alternatively, you can set the environment variable ``IPOPTWINDIR`` to point to
the Ipopt directory that contains the ``bin``, ``lib`` and ``include``
directories.

2. Using Conda Forge's IPOPT binary:

If using conda, you can install an IPOPT binary from Conda Forge::

   $ conda.exe install -c conda-forge ipopt

The environment variable ``IPOPTWINDIR`` should then be set to ``USECONDAFORGEIPOPT``.

Finally, execute::

   $ python setup.py install

**NOTE:** It is advised to use the Anaconda or Miniconda distributions and *not* the
official python.org distribution. Even though it has been tested to work with the
latest builds, it is well-known for causing issues. (see
https://github.com/mechmotum/cyipopt/issues/52).

On Ubuntu 18.04 Using APT Dependencies
--------------------------------------

All of the dependencies can be installed with Ubuntu's package manager::

   $ sudo apt install build-essential pkg-config python-dev cython python-numpy coinor-libipopt1v5 coinor-libipopt-dev

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

And finally install the package into Python's default package directory::

   $ python setup.py install

Note that you may or may not want to install this package system wide, i.e.
prepend ``sudo`` to the above command, but it is safest to install into your
user space, i.e. what ``pip install --user`` does, or setup a virtual
environment with tools like venv or conda. If you use virtual environments you
will need to be careful about selecting headers and libraries for packages in
or out of the virtual environments in the build step. Note that cython,
and numpy could alternatively be installed using Python specific package
managers, e.g. ``pip install cython numpy``.

On Ubuntu 18.04 with Custom Compiled IPOPT
------------------------------------------

Install system wide dependencies::

   $ sudo apt install pkg-config python-dev wget
   $ sudo apt build-dep coinor-libipopt1v5

Install ``pip`` so all Python packages can be installed via ``pip``::

   $ sudo apt install python-pip

Then use ``pip`` to install the following packages::

   $ pip install --user numpy cython

Compile Ipopt
~~~~~~~~~~~~~

The Ipopt compilation instructions are derived from
https://www.coin-or.org/Ipopt/documentation/node14.html. If you get errors,
start there for help.

Download Ipopt source code. Choose the version that you would like to have from
<https://www.coin-or.org/download/source/Ipopt/>. For example::

   $ cd ~
   $ wget https://www.coin-or.org/download/source/Ipopt/Ipopt-3.12.11.tgz

Extract the Ipopt source code::

   $ tar -xvf Ipopt-3.12.11.tgz

Create a temporary environment variable pointing to the Ipopt directory::

   $ export IPOPTDIR=~/Ipopt-3.12.11

To use linear solvers other than the default mumps, e.g. ``ma27, ma57, ma86``
solvers, the ``HSL`` package are needed. ``HSL`` can be downloaded from its
official website <http://www.hsl.rl.ac.uk/ipopt/>.

Extract ``HSL`` source code after you get it. Rename the extracted folder to
``coinhsl`` and copy it in the HSL folder: ``Ipopt-3.12.11/ThirdParty/HSL``

Build Ipopt::

   $ mkdir $IPOPTDIR/build
   $ cd $IPOPTDIR/build
   $ ../configure
   $ make
   $ make test

Add ``make install`` if you want a system wide install.

Set environment variables::

   $ export IPOPT_PATH="~/Ipopt-3.12.11/build"
   $ export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$IPOPT_PATH/lib/pkgconfig
   $ export PATH=$PATH:$IPOPT_PATH/bin

Get help from this web-page if you get errors in setting environments:

https://stackoverflow.com/questions/13428910/how-to-set-the-environmental-variable-ld-library-path-in-linux

Now compile ``cyipopt``. Download the ``cyipopt`` source code from PyPi, for
example::

   $ cd ~
   $ wget https://files.pythonhosted.org/packages/05/57/a7c5a86a8f899c5c109f30b8cdb278b64c43bd2ea04172cbfed721a98fac/ipopt-0.1.9.tar.gz
   $ tar -xvf ipopt-0.1.8.tar.gz
   $ cd ipopt

Compile ``cyipopt``::

   $ python setup.py build

If there is no error, then you have compiled ``cyipopt`` successfully

Check that everything linked correctly with ``ldd`` ::

   $ ldd build/lib.linux-x86_64-2.7/cyipopt.so
   linux-vdso.so.1 (0x00007ffe895e1000)
   libipopt.so.1 => /home/<username>/Ipopt-3.12.11/build/lib/libipopt.so.1 (0x00007f74efc2a000)
   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f74ef839000)
   libcoinmumps.so.1 => /home/<username>/Ipopt-3.12.11/build/lib/libcoinmumps.so.1 (0x00007f74ef4ae000)
   libcoinhsl.so.1 => /home/<username>/Ipopt-3.12.11/build/lib/libcoinhsl.so.1 (0x00007f74ef169000)
   liblapack.so.3 => /usr/lib/x86_64-linux-gnu/liblapack.so.3 (0x00007f74ee8cb000)
   libblas.so.3 => /usr/lib/x86_64-linux-gnu/libblas.so.3 (0x00007f74ee65e000)
   libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f74ee45a000)
   libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f74ee0d1000)
   libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f74edd33000)
   /lib64/ld-linux-x86-64.so.2 (0x00007f74f02c0000)
   libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f74edb1b000)
   libcoinmetis.so.1 => /home/<username>/Ipopt-3.12.11/build/lib/libcoinmetis.so.1 (0x00007f74ed8ca000)
   libgfortran.so.4 => /usr/lib/x86_64-linux-gnu/libgfortran.so.4 (0x00007f74ed4eb000)

Install ``cyipopt`` (prepend ``sudo`` if you want a system wide install)::

    $ python setup.py install

To use ``cyipopt`` you will need to set the ``LD_LIBRARY_PATH`` to point to
your Ipopt install if you did not install it to a standard location. For
example::

    $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/Ipopt-3.12.11/build/lib

You can add this to your shell's configuration file if you want it set every
time you open your shell, for example the following line can it can be added to
your ``~/.bashrc`` ::

    $ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/Ipopt-3.12.11/build/lib' >> ~/.bashrc

Now you should be able to run a ``cyipopt`` example::

    $ cd test
    $ python -c "import cyipopt"
    $ python examplehs071.py

If it could be run successfully, the optimization will start with the following
descriptions::

    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit http://projects.coin-or.org/Ipopt
    ******************************************************************************

    This is Ipopt version 3.12.11, running with linear solver ma27.
    ...

Conda Forge binaries with HSL
-----------------------------

It is possible to use the HSL linear solvers with cyipopt installed via Conda
Forge. To do so, first download the HSL source code tarball. The following
explanation uses ``coinhsl-2014.01.10.tar.gz`` with conda installed on Ubuntu
20.04.

Create a conda environment with at least gfortran and cyipopt::

   $ conda create -n hsl-test -c conda-forge gfortran cyipopt
   $ conda activate hsl-test

You should now have an environment that includes ipopt. You can checked what
ipopt is linked against like so::

   (hsl-test) $ ldd ~/miniconda/envs/hsl-test/lib/libipopt.so
      linux-vdso.so.1 (0x00007ffcaf45b000)
      librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007f8965748000)
      liblapack.so.3 => /home/<username>/miniconda/envs/hsl-test/lib/./liblapack.so.3 (0x00007f89635fe000)
      libdmumps_seq-5.2.1.so => /home/<username>/miniconda/envs/hsl-test/lib/./libdmumps_seq-5.2.1.so (0x00007f89633d8000)
      libmumps_common_seq-5.2.1.so => /home/<username>/miniconda/envs/hsl-test/lib/./libmumps_common_seq-5.2.1.so (0x00007f8963377000)
      libpord_seq-5.2.1.so => /home/<username>/miniconda/envs/hsl-test/lib/./libpord_seq-5.2.1.so (0x00007f896335e000)
      libmpiseq_seq-5.2.1.so => /home/<username>/miniconda/envs/hsl-test/lib/./libmpiseq_seq-5.2.1.so (0x00007f8963352000)
      libesmumps-6.so => /home/<username>/miniconda/envs/hsl-test/lib/./libesmumps-6.so (0x00007f8963349000)
      libscotch-6.so => /home/<username>/miniconda/envs/hsl-test/lib/./libscotch-6.so (0x00007f89632b1000)
      libscotcherr-6.so => /home/<username>/miniconda/envs/hsl-test/lib/./libscotcherr-6.so (0x00007f89632ac000)
      libmetis.so => /home/<username>/miniconda/envs/hsl-test/lib/./libmetis.so (0x00007f8963237000)
      libgfortran.so.5 => /home/<username>/miniconda/envs/hsl-test/lib/./libgfortran.so.5 (0x00007f896308e000)
      libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007f8963088000)
      libstdc++.so.6 => /home/<username>/miniconda/envs/hsl-test/lib/./libstdc++.so.6 (0x00007f8962edb000)
      libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f8962d8c000)
      libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f8962b9a000)
      /lib64/ld-linux-x86-64.so.2 (0x00007f8965a02000)
      libgcc_s.so.1 => /home/<username>/miniconda/envs/hsl-test/lib/./libgcc_s.so.1 (0x00007f8962b85000)
      libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f8962b62000)
      libgomp.so.1 => /home/<username>/miniconda/envs/hsl-test/lib/././libgomp.so.1 (0x00007f8962b2a000)
      libz.so.1 => /home/<username>/miniconda/envs/hsl-test/lib/././libz.so.1 (0x00007f8962b10000)
      libquadmath.so.0 => /home/<username>/miniconda/envs/hsl-test/lib/././libquadmath.so.0 (0x00007f8962ad6000)

Now navigate into the extracted HSL directory and configure HSL::

   (hsl-test) $ cd /path/to/coinhsl-2014.01.10/
   (hsl-test) $ ./configure \
      --prefix=/home/<username>/miniconda/envs/hsl-test/ \
      --with-blas="-L/home/<username>/miniconda/envs/hsl-test/lib/ -lblas" \
      LIBS="-llapack" \
      FC=/home/<username>/miniconda/envs/hsl-test/bin/gfortran \
      CC=/home/<username>/miniconda/envs/hsl-test/bin/gcc \

This tells HSL to install into your environment, link against the environment's
blas and lapack libraries and to use the environment's gfortran and gcc
compilers to build HSL. After configuring, build and install with::

   (hsl-test) $ make
   (hsl-test) $ make install

You should now find a shared HSL library in your environment. Check to make
sure it is properly linked (especially blas)::

   (hsl-test) $ ldd ~/miniconda/envs/hsl-test/lib/libcoinhsl.so
      linux-vdso.so.1 (0x00007ffe2085a000)
      libopenblas.so.0 => /home/<username>/miniconda/envs/hsl-test/lib/libopenblas.so.0 (0x00007f72a1766000)
      libgfortran.so.5 => /home/<username>/miniconda/envs/hsl-test/lib/libgfortran.so.5 (0x00007f72a15bd000)
      libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f72a143f000)
      libgcc_s.so.1 => /home/<username>/miniconda/envs/hsl-test/lib/libgcc_s.so.1 (0x00007f72a142a000)
      libquadmath.so.0 => /home/<username>/miniconda/envs/hsl-test/lib/libquadmath.so.0 (0x00007f72a13f0000)
      libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f72a11fe000)
      libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f72a11d9000)
      /lib64/ld-linux-x86-64.so.2 (0x00007f72a39d4000)

Now, in your cyipopt script set the following two options::

   problem.add_option('linear_solver', 'ma57')
   problem.add_option('hsllib', 'libcoinhsl.so')

The various HSL solvers can be set with ``linear_solver`` and the ``hsllib``
name must be specified because the default name ipopt looks for is
``libhsl.so``. Identify the shared library installed on your system and make
sure the name provided for the ``hsllib`` option matches. For example, on macOS
you may need ``problem.add_option('hsllib', 'libcoinhsl.dylib')``.
