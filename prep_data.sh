#!/bin/bash
set -e  

mkdir -p dataset
cd dataset

pip install gdown librosa speechbrain jiwer
gdown 19CV4WZgYez-i2oHV2r9maJofjNqcTX4o 
gdown 1v75mLO-TVfPXe27o54JMlXD5cQ81eaVG 
gdown 1YgTF-NbHuweHWr2LahS_X9j--laGDnIK

unzip -o voices.zip

cd /
python workspace/TASA/utils/construct.py
mkdir workspace/TASA/saves