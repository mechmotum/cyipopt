# -*- coding: utf-8 -*-
"""
cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012-2015 Amit Aides
Copyright (C) 2015-2017 Matthias Kümmerer
Copyright (C) 2017-2025 cyipopt developers

License: EPL 2.0
"""

import sys
import os.path
from distutils.sysconfig import get_python_lib
import subprocess as sp

from setuptools import setup
from setuptools.extension import Extension

# install requirements before import
# NOTE : NumPy 1.25 is the first release of NumPy that you can link to the
# NumPy C-API and it will be guaranteed to be backward compatible back to NumPy
# 1.19, thus we set the minimum NumPy build version to 1.25. See:
# https://numpy.org/doc/stable/dev/depending_on_numpy.html#adding-a-dependency-on-numpy
# for more information.
from setuptools import dist
SETUP_REQUIRES = [
    "cython>=0.29.37",
    "numpy>=1.25",
    "setuptools>=68.1.2",
]
dist.Distribution().fetch_build_eggs(SETUP_REQUIRES)

from Cython.Distutils import build_ext
import numpy as np


exec(open("cyipopt/version.py", encoding="utf-8").read())
PACKAGE_NAME = "cyipopt"
VERSION = __version__
DESCRIPTION = "A Cython wrapper to the IPOPT optimization package"
with open("README.rst", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()
KEYWORDS = [
    "coin-or",
    "interior-point",
    "ipopt",
    "nlp",
    "nonlinear programming",
    "optimization",
]
AUTHOR = "Jason K. Moore"
EMAIL = "moorepants@gmail.com"
URL = "https://github.com/mechmotum/cyipopt"
INSTALL_REQUIRES = [
    "numpy>=1.26.4",
]
LICENSE = "EPL-2.0"
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]


def pkgconfig(*packages, **kw):
    """Returns a dictionary containing the include and library flags to pass to
    a Python extension that depends on the provided packages.

    Parameters
    ----------
    *packages : one or more strings
        These are the names of ``.pc`` files that pkg-config can locate, for
        example ``ipopt`` for the ``ipopt.pc`` file.
    **kw : list of strings
        Any values that should be preset in the returned dictionary. These can
        include lists of items for: ``include_dirs``, ``library_dirs``,
        ``libraries``, ``extra_compile_args``.

    Returns
    -------
    kw : dictionary
        Dictionary containing the keys: ``include_dirs``, ``library_dirs``,
        ``libraries``, ``extra_compile_args`` that are mapped to lists of
        strings.

    Notes
    -----
    This function is based on:

    http://code.activestate.com/recipes/502261-python-distutils-pkg-config/#c2

    """
    flag_map = {"-I": "include_dirs", "-L": "library_dirs", "-l": "libraries"}
    output = sp.Popen(["pkg-config", "--libs", "--cflags"] + list(packages),
                      stdout=sp.PIPE).communicate()[0]

    if not output:  # output will be empty string if pkg-config finds nothing
        msg = ("pkg-config was not able to find any of the requested packages "
               "{} on your system. Make sure pkg-config can discover the .pc "
               "files associated with the installed packages.")
        raise OSError(msg.format(list(packages)))

    output = output.decode("utf8")

    for token in output.split():
        if token[:2] in flag_map:
            kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
        else:
            kw.setdefault("extra_compile_args", []).append(token)

    kw["include_dirs"] += [np.get_include()]

    return kw


def handle_ext_modules_win_32_other_ipopt():
    IPOPT_INCLUDE_DIRS = [os.path.join(ipoptdir, "include", "coin-or"),
                          np.get_include()]

    # These are the specific binaries in the IPOPT 3.13.2 binary download:
    # https://github.com/coin-or/Ipopt/releases/download/releases%2F3.13.2/Ipopt-3.13.2-win64-msvs2019-md.zip
    IPOPT_LIBS = ["ipopt.dll", "ipoptamplinterface.dll"]
    IPOPT_LIB_DIRS = [os.path.join(ipoptdir, "lib")]

    bin_folder = os.path.join(ipoptdir, "bin")
    IPOPT_DLL = [file for file in os.listdir(bin_folder) if file.endswith(".dll")]
    print("Found ipopt binaries {}".format(IPOPT_DLL))
    IPOPT_DLL_DIRS = [bin_folder]
    EXT_MODULES = [Extension("ipopt_wrapper",
                             ["cyipopt/cython/ipopt_wrapper.pyx"],
                             include_dirs=IPOPT_INCLUDE_DIRS,
                             libraries=IPOPT_LIBS,
                             library_dirs=IPOPT_LIB_DIRS)]
    DATA_FILES = [(get_python_lib(),
                  [os.path.join(IPOPT_DLL_DIRS[0], dll)
                   for dll in IPOPT_DLL])] if IPOPT_DLL else None
    include_package_data = False
    return EXT_MODULES, DATA_FILES, include_package_data


def handle_ext_modules_general_os():
    ipopt_wrapper_ext = Extension("ipopt_wrapper",
                                  ["cyipopt/cython/ipopt_wrapper.pyx"],
                                  **pkgconfig("ipopt"))
    EXT_MODULES = [ipopt_wrapper_ext]
    DATA_FILES = None
    include_package_data = True
    return EXT_MODULES, DATA_FILES, include_package_data


if __name__ == "__main__":

    ipoptdir = os.environ.get("IPOPTWINDIR", "")

    if sys.platform == "win32" and ipoptdir:
        print('Using Ipopt in {} directory on Windows.'.format(ipoptdir))
        ext_module_data = handle_ext_modules_win_32_other_ipopt()
    elif sys.platform == "win32" and not ipoptdir:
        try:  # first try adjacent files
            ipoptdir = os.path.abspath(os.path.dirname(__file__))
            msg = 'Using Ipopt adjacent to setup.py in {} on Windows.'
            print(msg.format(ipoptdir))
            ext_module_data = handle_ext_modules_win_32_other_ipopt()
        except FileNotFoundError:  # then look for ipopt.pc
            print('Using Ipopt found with pkg-config.')
            ext_module_data = handle_ext_modules_general_os()
    else:  # linux and mac
        print('Using Ipopt found with pkg-config.')
        ext_module_data = handle_ext_modules_general_os()
    EXT_MODULES, DATA_FILES, include_package_data = ext_module_data
    # NOTE : The `name` kwarg here is the distribution name, i.e. the name that
    # PyPi uses for a collection of packages. Historically this has been
    # `ipopt`, but as of 1.1.0 is `cyipopt`. `pip install cyipopt` will install
    # the `cyipopt` and `ipopt` packages into the `site-packages` directory.
    # Both `import cyipopt` and `import ipopt` will work, with the later giving
    # a deprecation warning.
    setup(name=PACKAGE_NAME,
          version=VERSION,
          author=AUTHOR,
          author_email=EMAIL,
          url=URL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          keywords=KEYWORDS,
          license=LICENSE,
          classifiers=CLASSIFIERS,
          packages=[PACKAGE_NAME],
          setup_requires=SETUP_REQUIRES,
          install_requires=INSTALL_REQUIRES,
          include_package_data=include_package_data,
          data_files=DATA_FILES,
          zip_safe=False,  # required for Py27 on Windows to work
          cmdclass={"build_ext": build_ext},
          ext_modules=EXT_MODULES,
          )
