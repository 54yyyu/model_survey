from layers import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv1D, MaxPooling1D,
                                     Dropout, Dense, 
                                     BatchNormalization,
                                     Activation, Flatten,
                                     LSTM, Bidirectional)
import numpy as np

def create(config, input_shape, output_shape):
    
    assert type(config) == dict
    
    num_conv_layers = config.nc
    
    
    
    if num_conv_layers==3:
        filters = [512, 256, 256]
        pool_sizes = [5, 4, 4]
        kernels = [19, 11, 7]
        dropouts = np.linspace(.1, .3, 3)
        
    elif num_conv_layers==4:
        filters = [512, 256, 256, 512]
        pool_sizes = [5, 2, 2, 4]
        kernels = [19, 11, 9, 7]
        dropouts = np.linspace(.1, .3, 4)
        
    else:
        raise Exception('c\'mon, be reasonable')
    
    inputs = keras.Inputs(input_shape)
    nn = inputs
    
    for layer, filter, pool, kernel, dropout in zip(range(num_conv_layers), filters, pool_sizes, kernels, dropouts):
        nn = Conv1D(filters=filter, kernel=kernel, padding='same')(nn)
        nn = BatchNormalization()(nn)
        if config.x1:
            nn = Conv1D(filters=filter, kernel=1, stride=1)(nn)
        nn = Activation('relu')(nn)
        if config.rb:
            nn = residual_block(nn, x1=config.x1)
        elif config.srb:
            nn = sot_residual_block(nn, x1=config.x1)
        nn = MaxPooling1D(pool)(nn)
        nn = Dropout(dropout)(nn)
    
    if config.LSTM:
        nn = Bidirectional(LSTM(96, return_sequences=True))(nn)
        #nn = Activation('relu')(nn)
        nn = Dropout(.2)(nn)
    
    for layer in range(config.nt):
        nn = transformer_block(nn)
    
    nn = Flatten()(nn)
    
    nn = Dense(1024)(nn)
    nn = BatchNormalization()(nn)
    nn = Activation('relu')(nn)
    nn = Dropout(.5)(nn)
    
    nn = Dense(1024)(nn)
    nn = BatchNormalization()(nn)
    nn = Activation('relu')(nn)
    nn = Dropout(.5)(nn)
    
    outputs = keras.layers.Dense(output_shape)(nn)
    
    return inputs, outputs