#!/bin/bash
# Builds manylinux cyipopt wheels with Ipopt 3.14.11 based on MUMPS 5.5.1, and OpenBLAS 0.3.15
set -eu  # Stop script if a line fails
TAG=${1}
echo "Building cyipopt with tag $TAG"

# Install efficient BLAS & LAPACK library
yum install -y openblas-devel-0.3.15-4.el8

pushd /tmp

# MUMPS (Linear solver used by Ipopt)
git clone https://github.com/coin-or-tools/ThirdParty-Mumps --depth=1 --branch releases/3.0.4
pushd ThirdParty-Mumps
sh get.Mumps
./configure --with-lapack="-L/usr/include/openblas -lopenblas"
make
make install
popd


# Ipopt (The solver itself)
git clone https://github.com/coin-or/Ipopt --depth=1 --branch releases/3.14.11
pushd Ipopt
./configure --with-lapack="-L/usr/include/openblas -lopenblas"
make
make install
popd

# build cyipopt for many python versions
git clone https://github.com/mechmotum/cyipopt --depth=1 --branch $TAG
pushd cyipopt
echo "------------------------------" >> LICENSE
echo "This binary distribution of cyipopt also bundles the following software" >> LICENSE
for bundled_license in licenses_manylinux_bundled_libraries/*.txt; do
    cat $bundled_license >> LICENSE
done
for PYVERSION in "cp311-cp311" "cp310-cp310" "cp39-cp39" "cp38-cp38" "pp39-pypy39_pp73" "pp38-pypy38_pp73" ; do
    /opt/python/$PYVERSION/bin/pip wheel --no-deps --wheel-dir=./dist .
done
for wheel in dist/cyipopt*; do
    auditwheel repair $wheel  # Inject solver's shared libraries to wheel
done
for wheel in wheelhouse/*.whl; do
    cp -rf $wheel /wheels/ # Copy repaired wheel to shared volume
done
popd

popd
