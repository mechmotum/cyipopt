==================
README for cyipopt
==================

Ipopt (Interior Point OPTimizer, pronounced eye-pea-Opt) is a software package
for large-scale nonlinear optimization.

cyipopt is a python wrapper around Ipopt. It enables using Ipopt from the
comfort of the great Python scripting language.

Ipopt is available from the `COIN-OR <https://projects.coin-or.org/Ipopt>`_
initiative, under the Eclipse Public License (EPL). 

Installing
==========

To install cyipopt you will need the following prerequisites:

  * python 2.6+
  * numpy
  * scipy
  * cython
  * C/C++ compiler

`Python(x,y) <http://code.google.com/p/pythonxy/>`_ is a great way to get all of
these if you are satisfied with 32bit Windows platform. Enthought's
`EPD <http://www.enthought.com/products/epd.php>`_ offers 32bit/64bit binaries
for Windows/Linux and OSX platforms.

You will also need the binaries and header files of the Ipopt package. I
recommend downloading the `binaries <http://www.coin-or.org/download/binary/Ipopt/>`_
especially as they include a version compiled against the MKL library.

Download the source files of cyipopt and update ``setup.py`` to point to the header
files and binaries of the Ipopt package.

Then, execute::

   python setup.py install


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

For bug reports use the Bitbucket issue tracker.
You can also send wishes, comments, patches, etc. to amitibo@tx.technion.ac.il


Acknowledgement
===============

Thank-you to the people at <http://wingware.com/> for their policy of **free licenses for non-commercial open source developers**.

.. image:: http://wingware.com/images/wingware-logo-180x58.png
