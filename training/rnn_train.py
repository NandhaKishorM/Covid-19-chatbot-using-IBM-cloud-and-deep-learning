#!/usr/bin/python

#from __future__ import print_function

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, GRU, SimpleRNN, Dropout, concatenate
from keras import losses, regularizers
from keras.constraints import min_max_norm, Constraint
from keras import backend as K
import numpy as np
import argparse, os
import h5py

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.42
#set_session(tf.Session(config=config))

def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)

#def my_accuracy(y_true, y_pred):
    #return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

def rnn_training(args):
    reg = 0.000001
    constraint = WeightClip(0.499)

    print('Build model...')
    main_input = Input(shape=(None, 42), name='main_input')
    tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
    vad_gru = GRU(24, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
    vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)
    noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
    noise_gru = GRU(48, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input)
    denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])

    denoise_gru = GRU(96, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input)

    denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(denoise_gru)

    model = Model(inputs=main_input, outputs=[denoise_output, vad_output])

    model.compile(loss=[mycost, my_crossentropy],
                  metrics=[msse],
                  optimizer='adam', loss_weights=[10, 0.5])

    batch_size = 32

    print('Loading data...')
    with h5py.File(args.data_file, 'r') as hf:
        all_data = hf['data'][:]
    print('done.')

    window_size = 2000

    nb_sequences = len(all_data)//window_size
    print(nb_sequences, ' sequences')
    x_train = all_data[:nb_sequences*window_size, :42]
    x_train = np.reshape(x_train, (nb_sequences, window_size, 42))

    y_train = np.copy(all_data[:nb_sequences*window_size, 42:64])
    y_train = np.reshape(y_train, (nb_sequences, window_size, 22))

    noise_train = np.copy(all_data[:nb_sequences*window_size, 64:86])
    noise_train = np.reshape(noise_train, (nb_sequences, window_size, 22))

    vad_train = np.copy(all_data[:nb_sequences*window_size, 86:87])
    vad_train = np.reshape(vad_train, (nb_sequences, window_size, 1))

    all_data = 0;
    #x_train = x_train.astype('float32')
    #y_train = y_train.astype('float32')

    print(len(x_train), 'train sequences. x shape =', x_train.shape, 'y shape = ', y_train.shape)

    print('Train...')
    model.fit(x_train, [y_train, vad_train],
              batch_size=batch_size,
              epochs=120,
              validation_split=0.1)
    model.save(args.model_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', help='feature matrix h5 data file', type=str, default=os.path.join(os.path.dirname(__file__), 'denoise_data.h5'))
    parser.add_argument('--model_file', help='output h5 model file', type=str, default=os.path.join(os.path.dirname(__file__), 'model_weights.hdf5'))
    args = parser.parse_args()
    if not args.data_file:
        raise ValueError('data file is missing')

    rnn_training(args)


if __name__ == "__main__":
    main()
