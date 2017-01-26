.. highlight:: sh

Installation
============

To install cyipopt you will need the following prerequisites:

* C/C++ compiler
* python 2.7 and 3.4+
* setuptools
* numpy
* cython
* six
* future
* scipy [optiional]

The `Anaconda Python Distribution <https://www.continuum.io/why-anaconda>`_ is
one of the easiest ways to install python for Linux, Mac and Windows.

You will also need the binaries and header files of the Ipopt package. I
recommend downloading the `binaries <http://www.coin-or.org/download/binary/Ipopt/>`_
especially as they include a version compiled against the MKL library.

Download the source files of cyipopt and update ``setup.py`` to point to the header
files and binaries of the Ipopt package, if `LD_LIBRARY_PATH` and `pkg_config` are
not setup to find ipopt on their own.

Then, execute::

   python setup.py install

You can test the installation by running the examples under the folder ``test/``

.. note::

    Under linux you might need to let the OS know where to look for the Ipopt lib files,
    e.g. use::

        $ export LD_LIBRARY_PATH=<PATH to Ipopt lib files>

