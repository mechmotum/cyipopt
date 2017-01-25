# -*- coding: utf-8 -*-
"""
cyipot: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012 Amit Aides, 2015 Matthias Kümmerer
Author: Matthias Kümmerer <matthias.kuemmerer@bethgelab.org>
(originally Author: Amit Aides <amitibo@tx.technion.ac.il>)
URL: <https://bitbucket.org/amitibo/cyipopt>
License: EPL 1.0
"""
from setuptools import setup
from setuptools.extension import Extension
from distutils.sysconfig import get_python_lib
from Cython.Distutils import build_ext
import Cython.Distutils
import Cython.Compiler.Options
import numpy as np
import os.path
import sys
import six


PACKAGE_NAME = 'ipopt'
VERSION = '0.1.7'
DESCRIPTION = 'A Cython wrapper to the IPOPT optimization package'
AUTHOR = 'Matthias Kümmerer'
EMAIL = 'matthias.kuemmerer@bethgelab.org'
URL = "https://github.com/matthiask/cyipopt"



def main_win32():
    IPOPT_ICLUDE_DIRS=['include_mt/coin', np.get_include()]
    IPOPT_LIBS=['Ipopt39', 'IpoptFSS']
    IPOPT_LIB_DIRS=['lib_mt/x64/release']
    IPOPT_DLL=['Ipopt39.dll', 'IpoptFSS39.dll']

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


def pkgconfig(*packages, **kw):
    """Based on http://code.activestate.com/recipes/502261-python-distutils-pkg-config/#c2"""

    import subprocess as sp

    flag_map = {'-I': 'include_dirs', '-L': 'library_dirs', '-l': 'libraries'}
    output = sp.Popen(["pkg-config", "--libs", "--cflags"] + list(packages), stdout=sp.PIPE).communicate()[0]
    if six.PY3:
        output = output.decode('utf8')
    for token in output.split():
        if token[:2] in flag_map:
            kw.setdefault(flag_map.get(token[:2]), []).append(token[2:])
        else:
            kw.setdefault('extra_compile_args', []).append(token)

    kw['include_dirs'] += [np.get_include()]

    return kw


def main_unix():
    setup(name = PACKAGE_NAME,
          version=VERSION,
          packages = [PACKAGE_NAME],
          cmdclass = {'build_ext': Cython.Distutils.build_ext},
          include_package_data=True,
          ext_modules = [Extension("cyipopt",
                                   ['src/cyipopt.pyx'],
                                   **pkgconfig('ipopt')
                                   )
                        ]
         )


if __name__ == '__main__':
    if sys.platform == 'win32':
        main_win32()
    else:
        main_unix()
