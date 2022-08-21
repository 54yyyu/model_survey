from .. import layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv1D, MaxPooling1D,
                                     Dropout, Dense, 
                                     BatchNormalization,
                                     Activation, Flatten,
                                     LSTM, Bidirectional)
import numpy as np

def create(config, input_shape, output_shape):
    
    dataset = config.dataset
    
        

    if dataset == 'deepstar':
        filters = [256, 128]
    if dataset == 'basset':
        filters = [300, 200]
    if dataset == 'GM':
        filters = [768, 512]

    pool_sizes = [None, 10]
    kernels = [19, 7]
    dropouts = [.1, .1]
    
    
    inputs = keras.Input(input_shape)
    nn = inputs
    
    for filter, pool, kernel, dropout in zip(filters, pool_sizes, kernels, dropouts):
        nn = Conv1D(filters=filter, kernel_size=kernel, padding='same')(nn)
        nn = BatchNormalization()(nn)
        if config.x1:
            nn = Conv1D(filters=filter, kernel_size=1, strides=1)(nn)
        nn = Activation('relu')(nn)
        if pool is not None:    
            nn = MaxPooling1D(pool)(nn)
        nn = Dropout(dropout)(nn)
    
    if config.LSTM:
        nn = Bidirectional(LSTM(96, return_sequences=True))(nn)
        #nn = Activation('relu')(nn)
        nn = Dropout(.2)(nn)
    
    for layer in range(config.nt):
        nn = layers.transformer_block(nn)
    
    representation = layers.LayerNormalization(epsilon=1e-5)(nn)
    attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    nn = tf.squeeze(weighted_representation, -2)
    
    if dataset == 'deepstar':
        nn = Dense(256)(nn)
        nn = BatchNormalization()(nn)
        nn = Activation('relu')(nn)
        nn = Dropout(.5)(nn)
        
        nn = Dense(256)(nn)
        nn = BatchNormalization()(nn)
        nn = Activation('relu')(nn)
        nn = Dropout(.5)(nn)
    else:
        nn = Dense(1024)(nn)
        nn = BatchNormalization()(nn)
        nn = Activation('relu')(nn)
        nn = Dropout(.5)(nn)
        
        nn = Dense(1024)(nn)
        nn = BatchNormalization()(nn)
        nn = Activation('relu')(nn)
        nn = Dropout(.5)(nn)
    
    outputs = keras.layers.Dense(output_shape, activation='sigmoid')(nn)
    
    return inputs, outputs