#!/bin/bash

set -e
mkdir -p /tmp/cyipopt && cd /tmp/cyipopt
git clone https://github.com/matthias-k/cyipopt
cd cyipopt
python setup.py install
cd test
python -c "import ipopt"
python examplehs071.py
cd /tmp && rm -rf cyipopt
