#!/usr/bin/python

import argparse
import os
import numpy as np
import h5py

def file_trans(args):
    height, width = args.matrix_shape.split('x')
    data = np.fromfile(args.bin_file, dtype='float32');
    data = np.reshape(data, (int(height), int(width)));
    h5f = h5py.File(args.h5_file, 'w');
    h5f.create_dataset('data', data=data)
    h5f.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_file', help='feature matrix bin file generated with denoise_training', type=str)
    parser.add_argument('--matrix_shape', help='feature matrix shape as <num>x<num>', type=str)
    parser.add_argument('--h5_file', help='output h5 file', type=str, default=os.path.join(os.path.dirname(__file__), 'denoise_data.h5'))
    args = parser.parse_args()
    if not args.bin_file:
        raise ValueError('bin file is missing')
    if not args.matrix_shape:
        raise ValueError('matrix shape is missing')

    file_trans(args)


if __name__ == "__main__":
    main()
