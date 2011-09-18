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

PACKAGE_NAME = 'ipopt'
IPOPT_ICLUDE_DIR=r'include\coin'
IPOPT_LIB='Ipopt'
IPOPT_LIB_DIR=r'lib\win32\Release MKL'
#IPOPT_LIB_DIR=r'lib\win32\Debug'
IPOPT_DLL='Ipopt39.dll'

setup(
    name=PACKAGE_NAME,
    version='0.1',
    description='A Cython wrapper to the IPOPT optimization package',
    author='Amit Aides',
    author_email='amitibo@tx.technion.ac.il',
    url="",
    packages=[PACKAGE_NAME],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension(
            PACKAGE_NAME + '.' + 'cyipopt',
            [r'src\cyipopt.pyx'],
            include_dirs=[IPOPT_ICLUDE_DIR, np.get_include()],
            libraries=[IPOPT_LIB],
            library_dirs=[IPOPT_LIB_DIR]
        )
    ],
    data_files=[(os.path.join(get_python_lib(), PACKAGE_NAME), [os.path.join(IPOPT_LIB_DIR, IPOPT_DLL)])]
)