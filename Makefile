# cyipot: Python wrapper for the Ipopt optimization package, written in Cython.
#
# Copyright (C) 2012 Amit Aides
# Author: Amit Aides <amitibo@tx.technion.ac.il>
# URL: <http://http://code.google.com/p/cyipopt/>
# License: EPL 1.0


PYTHON = python

exec_prefix = ${prefix}
prefix      = /home/amitibo/code/Ipopt-3.10.1
libdir      = ${exec_prefix}/lib

CXX         = g++
CXXFLAGS    = -O3 -pipe -DNDEBUG -pedantic-errors -Wparentheses -Wreturn-type -Wcast-qual -Wall -Wpointer-arith -Wwrite-strings -Wconversion -Wno-unknown-pragmas -Wno-long-long   -DIPOPT_BUILD -DMATLAB_MEXFILE # -DMWINDEXISINT
LDFLAGS     = $(CXXFLAGS)  -Wl,--rpath -Wl,/home/amitibo/code/Ipopt-3.10.1/lib

# Include directories (we use the CYGPATH_W variables to allow compilation with Windows compilers)
INCL = `PKG_CONFIG_PATH=/home/amitibo/code/Ipopt-3.10.1/lib/pkgconfig:/home/amitibo/code/Ipopt-3.10.1/share/pkgconfig: /usr/bin/pkg-config --cflags ipopt`
#INCL = -I`$(CYGPATH_W) /home/amitibo/code/Ipopt-3.10.1/include/coin` 

# Linker flags
LIBS = `PKG_CONFIG_PATH=/home/amitibo/code/Ipopt-3.10.1/lib/pkgconfig:/home/amitibo/code/Ipopt-3.10.1/share/pkgconfig: /usr/bin/pkg-config --libs ipopt`
##LIBS = -link -libpath:`$(CYGPATH_W) /home/amitibo/code/Ipopt-3.10.1/lib` libipopt.lib -lm  -ldl
#LIBS = -L/home/amitibo/code/Ipopt-3.10.1/lib -lipopt -lm  -ldl

# The following is necessary under cygwin, if native compilers are used
CYGPATH_W = echo


.PHONY: usage all clean clean_code

usage:
	@echo "make dist -- Build distributions (output to dist/)"
	@echo "make clean -- Remove all built files and temporary files"

all: dist

########################################################################
# DISTRIBUTIONS
########################################################################

dist: zipdist gztardist windist

gztardist: clean_code
	$(PYTHON) setup.py -q sdist --format=gztar
zipdist: clean_code
	$(PYTHON) setup.py -q sdist --format=zip
rpmdist: clean_code
	$(PYTHON) setup.py -q bdist --format=rpm
windist: clean_code
	$(PYTHON) setup.py -q bdist --format=wininst --plat-name=win32

########################################################################
# CLEAN
########################################################################

.PHONY: clean clean_code

clean:
	rm -rf build dist MANIFEST

clean_code:
	rm -f `find . -name '*.pyc'`
	rm -f `find . -name '*.pyo'`
	rm -f `find . -name '*~'`
	rm -f MANIFEST # regenerate manifest from MANIFEST.in