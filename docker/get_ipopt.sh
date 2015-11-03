#!/bin/bash

set -e
mkdir -p /tmp/ipopt
cd /tmp/ipopt
wget -c http://www.coin-or.org/download/source/Ipopt/Ipopt-3.12.4.zip
unzip Ipopt-3.12.4.zip && cd Ipopt-3.12.4/
export IPOPTDIR=/tmp/ipopt/Ipopt-3.12.4/
cd $IPOPTDIR/ThirdParty/Blas \
./get.Blas
cd ../Lapack && ./get.Lapack
cd ../ASL && ./get.ASL
cd ../Mumps && ./get.Mumps 
mkdir $IPOPTDIR/build
cd $IPOPTDIR/build
$IPOPTDIR/configure --prefix=/opt/ipopt/
make
make test
make install
echo "/opt/ipopt/lib" > /etc/ld.so.conf.d/ipopt.conf; ldconfig;
echo 'export IPOPTPATH="/opt/ipopt"' >> /etc/bash.basrc
echo 'export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$IPOPTPATH/lib/pkgconfig' >> /etc/bash.bashrc
echo 'export PATH=$PATH:$IPOPTPATH/bin' >> /etc/bash.bashrc
cd /tmp
rm -rf ipopt
