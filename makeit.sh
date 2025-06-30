#!/bin/bash

./clean.sh
pip install -r requirements.txt
python setup.py build_ext --inplace
python setup.py install
tar -xf test_instrument.tar -C $SIXTE/share/sixte/instruments
