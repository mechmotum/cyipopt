# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 12:54:15 2011

@author: amitibo
"""
from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_lib
from Cython.Distutils import build_ext
import numpy as np
import os.path
import sys


PACKAGE_NAME = 'ipopt'

if sys.version == 'win32':
    IPOPT_ICLUDE_DIRS=['include/coin']
    IPOPT_LIBS=['Ipopt']
    IPOPT_LIB_DIRS=['lib/win32/Release MKL']
    IPOPT_DLL='Ipopt39.dll'
else:
    IPOPT_ICLUDE_DIRS=['/usr/include/coin']
    #IPOPT_LIBS=['ipopt', 'coinhsl', 'coinlapack', 'coinblas', 'coinmumps', 'coinmetis']
    IPOPT_LIBS=['ipopt']
    #IPOPT_LIB_DIRS=['/home/hschilli/local/lib/coin', '/home/hschilli/local/lib/coin/ThirdParty']
    IPOPT_LIB_DIRS=['/usr/lib']
    IPOPT_DLL='libipopt.so'

IPOPT_ICLUDE_DIRS += [np.get_include()]

setup(
    name=PACKAGE_NAME,
    version='0.1.1',
    description='A Cython wrapper to the IPOPT optimization package',
    author='Amit Aides',
    author_email='amitibo@tx.technion.ac.il',
    url="",
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
    data_files=[(os.path.join(get_python_lib(), PACKAGE_NAME), [os.path.join(IPOPT_LIB_DIRS[0], IPOPT_DLL)])]
)
