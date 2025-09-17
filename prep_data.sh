#!/bin/bash

if [[ "$1" == "--help" ]]; then
    echo "How to: $0 [option]"
    echo "  --help     Instructions"
    echo "  phoneme    Preprocess for phoneme-based model"
    echo "  normal     Preprocess for normal model (default)"
    exit 0
fi

set -e  

mkdir -p dataset
cd dataset

pip install gdown librosa speechbrain jiwer
gdown 19CV4WZgYez-i2oHV2r9maJofjNqcTX4o 
gdown 1v75mLO-TVfPXe27o54JMlXD5cQ81eaVG 
gdown 1YgTF-NbHuweHWr2LahS_X9j--laGDnIK

unzip -o voices.zip

cd /
if [[ "$1" == "phoneme" ]]; then
    echo "Preprocessing for phoneme-based model"
    python /home/anhkhoa/TASA/utils/construct_phoneme.py
else
    echo "Preprocessing for normal model"
    python workspace/TASA/utils/construct.py
fi
mkdir workspace/TASA/saves