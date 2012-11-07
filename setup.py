# -*- coding: utf-8 -*-
"""
cyipot: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012 Amit Aides
Author: Amit Aides <amitibo@tx.technion.ac.il>
URL: <http://http://code.google.com/p/cyipopt/>
License: EPL 1.0
"""
from setuptools import setup
from setuptools.extension import Extension
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

if sys.platform == 'win32':
    IPOPT_ICLUDE_DIRS=['include_mt/coin']
    IPOPT_LIBS=['Ipopt39', 'IpoptFSS']
    IPOPT_LIB_DIRS=['lib_mt/x64/release']
    IPOPT_DLL=['Ipopt39.dll', 'IpoptFSS39.dll']
else:
    IPOPT_ICLUDE_DIRS=['/u/amitibo/.local/include/coin', '/usr/include/atlas-x86_64-base']
    IPOPT_LIBS=['ipopt', 'coinasl', 'atlas', 'cblas', 'clapack', 'coinmumps', 'coinmetis']
    IPOPT_LIB_DIRS=['/u/amitibo/.local/lib', '/usr/lib64/atlas']
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
        data_files=[(os.path.join(get_python_lib(), PACKAGE_NAME), [os.path.join(IPOPT_LIB_DIRS[0], dll) for dll in IPOPT_DLL])] if IPOPT_DLL else None
    )


if __name__ == '__main__':
    main()
