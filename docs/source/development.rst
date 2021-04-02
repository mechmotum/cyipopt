===========
Development
===========

Building the documentation
==========================

First, install sphinx, e.g.::

   conda install sphinx

Then, after installing cyipopt, navigate to a directory that contains the
source code and execute::

   $ cd docs
   $ make html

Once the build process finishes, direct your web browser to
``build/html/index.html``.

Testing
=======

You can test the installation by running each of the examples in the ``examples/`` directory.

If you're a developer, to properly run the packages' test suite you will need to make sure you have ``pytest`` installed. This can be done with::

    $ pip install pytest

if you are using a Python ``venv``, or with::

    $ conda install pytest

if you have a ``conda`` virtual environment set up. The tests can then run by calling::

    $ pytest

