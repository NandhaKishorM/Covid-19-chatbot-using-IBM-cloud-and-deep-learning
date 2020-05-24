#!/bin/sh
#
# run this script to generate rnnoise audio feature
# extraction tool for training

gcc -DTRAINING=1 -Wall -W -O3 -g -I../include denoise.c pitch.c celt_lpc.c rnn.c rnn_data.c -o denoise_training -lm
