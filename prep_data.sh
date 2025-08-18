#!/bin/bash
set -e  

mkdir -p dataset
cd dataset

command -v gdown >/dev/null 2>&1 || pip install gdown
gdown 19CV4WZgYez-i2oHV2r9maJofjNqcTX4o 
gdown 1v75mLO-TVfPXe27o54JMlXD5cQ81eaVG 
gdown 1YgTF-NbHuweHWr2LahS_X9j--laGDnIK

unzip -o voices.zip

cd ..
python TASA/utils/construct.py
