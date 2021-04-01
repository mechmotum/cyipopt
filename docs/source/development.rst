Building the Documentation
==========================

After installing::

   cd docs
   make html

Then, direct your browser to ``build/html/index.html``.

Testing
=======

You can test the installation by running each of the examples in the
``examples/`` directory.

If you're a developer, to properly run the packages' test suite you will need
to make sure you have ``pytest`` installed. This can be done with::

    $ pip install pytest

if you are using a Python ``venv``, or with::

    $ conda install pytest

if you have a ``conda`` environment set up. The tests can then run by calling::

    $ pytest
