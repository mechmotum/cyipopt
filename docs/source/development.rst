.. highlight:: sh

===========
Development
===========

Development Install
===================

Clone the repository::

   $ git clone git@github.com:mechmotum/cyipopt.git
   $ cd cyipopt

Create a Conda environment with the dependencies::

   $ conda env create -f conda/cyipopt-dev.yml

Activate the environment::

   $ conda activate cyipopt-dev

Install a development version [1]_::

   (cyipopt-dev)$ python setup.py develop

.. [1] Changes to any of the Cython files require calling ``python setup.py
   develop`` to see effects of the changes.

Building the documentation
==========================

After installing the development version of cyipopt, navigate to a directory
that contains the source code and execute the ``Makefile``::

   (cyipopt-dev)$ cd docs
   (cyipopt-dev)$ make html

Once the build process finishes, direct your web browser to
``build/html/index.html``.

Testing
=======

You can test the installation by running each of the examples in the
``examples/`` directory and running the test suite. The tests can be run with::

    (cyipopt-dev)$ pytest
