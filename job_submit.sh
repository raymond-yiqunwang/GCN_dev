#!/bin/bash

TARGET="energy_per_atom"
#RESUME="pre-trained/final-energy-per-atom.pth.tar"

#TARGET="formation_energy_per_atom"
#RESUME="pre-trained/formation-energy-per-atom.pth.tar"

#TARGET="band_gap"
#RESUME="pre-trained/band-gap.pth.tar"

python main.py \
    --target $TARGET \
    --resume $RESUME \
    --h-fea-len 32 \
    --n-conv 4
