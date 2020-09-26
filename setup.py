#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012-2015 Amit Aides
Copyright (C) 2015-2017 Matthias Kümmerer
Copyright (C) 2017-2020 cyipopt developers

Author: Matthias Kümmerer <matthias.kuemmerer@bethgelab.org>
(original Author: Amit Aides <amitibo@tx.technion.ac.il>)
URL: https://github.com/matthias-k/cyipopt
License: EPL 1.0
"""

import sys
import os.path
from distutils.sysconfig import get_python_lib
import subprocess as sp

from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import six

if six.PY3:
    exec(open("cyipopt/version.py", encoding="utf-8").read())
else:
    exec(open("cyipopt/version.py").read())

PACKAGE_NAME = "cyipopt"
DEPRICATED_PACKAGE_NAME = "ipopt"
VERSION = __version__
DESCRIPTION = "A Cython wrapper to the IPOPT optimization package"
if six.PY3:
    with open("README.rst", encoding="utf-8") as f:
        LONG_DESCRIPTION = f.read()
else:
    with open("README.rst") as f:
        LONG_DESCRIPTION = f.read()
KEYWORDS = [
    "optimization",
    "nonlinear programming",
    "ipopt",
    "interior-point",
    "nlp",
    "coin-or",
]
AUTHOR = "Matthias Kümmerer"
EMAIL = "matthias.kuemmerer@bethgelab.org"
URL = "https://github.com/matthias-k/cyipopt"
DEPENDENCIES = [
    "numpy",
    "cython",
    "six",
    "future",
    "setuptools",
]
LICENSE = "EPL-1.0"
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: Eclipse Public License 1.0 (EPL-1.0)",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 2.7",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]


def pkgconfig(*packages, **kw):
    """Based on http://code.activestate.com/recipes/502261-python-distutils-pkg-config/#c2"""

    flag_map = {"-I": "include_dirs", "-L": "library_dirs", "-l": "libraries"}
    output = sp.Popen(["pkg-config", "--libs", "--cflags"] + list(packages),
                      stdout=sp.PIPE).communicate()[0]

    if not output:  # output will be an empty string if pkg-config finds nothing
        msg = ("pkg-config was not able to find any of the requested packages "
               "{} on your system. Make sure pkg-config can discover the .pc "
               "files associated with the installed packages.")
        raise OSError(msg.format(list(packages)))

    if six.PY3:
        output = output.decode("utf8")
    for token in output.split():
        if token[:2] in flag_map:
            kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
        else:
            kw.setdefault("extra_compile_args", []).append(token)

    kw["include_dirs"] += [np.get_include()]

    return kw


def handle_ext_modules_win_32():
    ipoptdir = os.environ.get("IPOPTWINDIR", "")
    IPOPT_INCLUDE_DIRS = [os.path.join(ipoptdir, "include", "coin-or"),
                          np.get_include()]
    IPOPT_LIBS = ["ipopt.dll", "ipoptamplinterface.dll"]
    IPOPT_LIB_DIRS = [os.path.join(ipoptdir, "lib")]
    IPOPT_DLL = ["ipopt-3.dll", "ipoptamplinterface-3.dll", "libifcoremd.dll",
                 "libmmd.dll", "msvcp140.dll", "svml_dispmd.dll",
                 "vcruntime140.dll"]
    IPOPT_DLL_DIRS = [os.path.join(ipoptdir, "bin")]
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

    if sys.platform == "win32":
        ext_module_data = handle_ext_modules_win_32()
    else:
        ext_module_data = handle_ext_modules_general_os()
    EXT_MODULES, DATA_FILES, include_package_data = ext_module_data
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        keywords=KEYWORDS,
        license=LICENSE,
        classifiers=CLASSIFIERS,
        packages=[PACKAGE_NAME, DEPRICATED_PACKAGE_NAME],
        install_requires=DEPENDENCIES,
        include_package_data=include_package_data,
        data_files=DATA_FILES,
        zip_safe=False,  # required for Py27 on Windows to work
        cmdclass={"build_ext": build_ext},
        ext_modules=EXT_MODULES,
    )
