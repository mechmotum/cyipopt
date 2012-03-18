# -*- coding: utf-8 -*-
"""
cyipot: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012 Amit Aides
Author: Amit Aides <amitibo@tx.technion.ac.il>
URL: <http://http://code.google.com/p/cyipopt/>
License: EPL 1.0
"""
from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_lib
from Cython.Distutils import build_ext
import numpy as np
import os.path
import sys


PACKAGE_NAME = 'ipopt'
VERSION = '0.1.3'
DESCRIPTION = 'A Cython wrapper to the IPOPT optimization package'
AUTHOR = 'Amit Aides'
EMAIL = 'amitibo@tx.technion.ac.il'
URL = "http://code.google.com/p/cyipopt/"

if sys.version == 'win32':
    IPOPT_ICLUDE_DIRS=['include/coin']
    IPOPT_LIBS=['Ipopt']
    IPOPT_LIB_DIRS=['lib/win32/Release MKL']
    IPOPT_DLL='Ipopt39.dll'
else:
    IPOPT_ICLUDE_DIRS=['/home/amitibo/code/Ipopt-3.10.1/include/coin']
    IPOPT_LIBS=['ipopt', 'coinhsl', 'coinlapack', 'coinblas', 'coinmumps', 'coinmetis']
    IPOPT_LIB_DIRS=['/home/amitibo/code/Ipopt-3.10.1/lib']
    IPOPT_DLL=None

IPOPT_ICLUDE_DIRS += [np.get_include()]

def main():
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        packages=[PACKAGE_NAME],
        cmdclass = {'build_ext': build_ext},
        ext_modules = [
            Extension(
                PACKAGE_NAME + '.' + 'cyipopt',
                ['src/cyipopt.pyx'],
                include_dirs=IPOPT_ICLUDE_DIRS,
                libraries=IPOPT_LIBS,
                library_dirs=IPOPT_LIB_DIRS
            )
        ],
        data_files=[(os.path.join(get_python_lib(), PACKAGE_NAME), [os.path.join(IPOPT_LIB_DIRS[0], IPOPT_DLL)])] if IPOPT_DLL else None
    )


if __name__ == '__main__':
    main()
