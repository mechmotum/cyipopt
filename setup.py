#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012-2015 Amit Aides
Copyright (C) 2015-2018 Matthias Kümmerer

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
import Cython.Distutils
import Cython.Compiler.Options
import numpy as np
import six

if six.PY3:
    exec(open('ipopt/version.py', encoding="utf-8").read())
else:
    exec(open('ipopt/version.py').read())

PACKAGE_NAME = 'ipopt'
VERSION = __version__
DESCRIPTION = 'A Cython wrapper to the IPOPT optimization package'
if six.PY3:
    with open('README.rst', encoding="utf-8") as f:
        LONG_DESCRIPTION = f.read()
else:
    with open('README.rst') as f:
        LONG_DESCRIPTION = f.read()
AUTHOR = 'Matthias Kümmerer'
EMAIL = 'matthias.kuemmerer@bethgelab.org'
URL = "https://github.com/matthias-k/cyipopt"
DEPENDENCIES = ['numpy', 'cython', 'six', 'future', 'setuptools']


def pkgconfig(*packages, **kw):
    """Based on http://code.activestate.com/recipes/502261-python-distutils-pkg-config/#c2"""

    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    output = sp.Popen(["pkg-config", "--libs", "--cflags"] + list(packages),
                      stdout=sp.PIPE).communicate()[0]
    if six.PY3:
        output = output.decode('utf8')
    for token in output.split():
        if token[:2] in flag_map:
            kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
        else:
            kw.setdefault('extra_compile_args', []).append(token)

    kw['include_dirs'] += [np.get_include()]

    return kw


if __name__ == '__main__':

    if sys.platform == 'win32':

        IPOPT_INCLUDE_DIRS = ['include_mt/coin', np.get_include()]
        IPOPT_LIBS = ['Ipopt39', 'IpoptFSS']
        IPOPT_LIB_DIRS = ['lib_mt/x64/release']
        IPOPT_DLL = ['Ipopt39.dll', 'IpoptFSS39.dll']

        EXT_MODULES = [
            Extension(
                PACKAGE_NAME + '.' + 'cyipopt',
                ['src/cyipopt.pyx'],
                include_dirs=IPOPT_INCLUDE_DIRS,
                libraries=IPOPT_LIBS,
                library_dirs=IPOPT_LIB_DIRS
            )
        ]
        DATA_FILES = [(os.path.join(get_python_lib(), PACKAGE_NAME),
                      [os.path.join(IPOPT_LIB_DIRS[0], dll)
                      for dll in IPOPT_DLL])] if IPOPT_DLL else None
        include_package_data = False

    else:

        EXT_MODULES = [Extension("cyipopt", ['src/cyipopt.pyx'],
                                 **pkgconfig('ipopt'))]
        DATA_FILES = None
        include_package_data = True

    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        keywords="optimization",
        license="EPL-1.0",
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Eclipse Public License 1.0 (EPL-1.0)',
            'Intended Audience :: Science/Research',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            ],
        packages=[PACKAGE_NAME],
        install_requires=DEPENDENCIES,
        include_package_data=include_package_data,
        data_files=DATA_FILES,
        cmdclass={'build_ext': build_ext},
        ext_modules=EXT_MODULES
    )
