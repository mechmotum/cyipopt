.. highlight:: sh

Installation
============

To install cyipopt you will need the following prerequisites:

* python 2.6+
* numpy
* scipy
* cython

`Python(x,y) <http://code.google.com/p/pythonxy/>`_ is a great way to get all
of these if you are using windows and satisfied with 32bit.

You will also need the binaries and header files of the Ipopt package. I
recommend downloading the binaries from http://www.coin-or.org/download/binary/Ipopt/
especially as they include a version compiled against the MKL library.

Download the source files of cyipopt and update ``setup.py`` to point to the
header files and binaries of the Ipopt package. Then, execute::

    $ python setup.py install

You can test the installation by running the examples under the folder ``test/``

.. note::

    Under linux you might need to let the OS know where to look for the Ipopt lib files,
    e.g. use::

        $ export LD_LIBRARY_PATH=<PATH to Ipopt lib files>

