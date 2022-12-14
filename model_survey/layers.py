import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


def residual_block(input_layer, kernel_size=3, activation='relu', num_layers=3, dropout=0.1, x1=False):

    filters = input_layer.shape.as_list()[-1]  

    nn = keras.layers.Conv1D(filters=filters,
                            kernel_size=kernel_size,
                            activation=None,
                            use_bias=False,
                            padding='same',
                            dilation_rate=1)(input_layer) 
    nn = keras.layers.BatchNormalization()(nn)
    if x1:
        nn = keras.layers.Conv1D(filters=filters,
                                 kernel_size=1,
                                 strides=1,
                                 activation=None)

    base_rate = 2
    for i in range(num_layers):
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Conv1D(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=1,
                                    padding='same',
                                    dilation_rate=base_rate**i)(nn) 
        nn = keras.layers.BatchNormalization()(nn)
        if x1:
            nn = keras.layers.Conv1D(filters=filters,
                                     kernel_size=1,
                                     strides=1,
                                     activation=None)
    nn = keras.layers.Add()([input_layer, nn])
    return keras.layers.Activation(activation)(nn)


def sot_residual_block(input_layer, kernel_size=3, activation='relu', num_layers=3, dropout=0.1, x1=False):

    filters = input_layer.shape.as_list()[-1]  

    nn = keras.layers.Conv1D(filters=filters,
                            kernel_size=kernel_size,
                            activation=None,
                            use_bias=False,
                            padding='same')(input_layer) 
    nn = keras.layers.BatchNormalization()(nn)

    if x1:
        nn = keras.layers.Conv1D(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 activation=None)(nn)
    
    for i in range(num_layers):
        nn = keras.layers.Activation('relu')(nn)
        nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Conv1D(filters=filters,
                                    kernel_size=kernel_size,
                                    strides=1,
                                    padding='same',)(nn) 
        nn = keras.layers.BatchNormalization()(nn)
        if x1:
            nn = keras.layers.Conv1D(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=1,
                                     activation=None)
    nn = tfa.layers.StochasticDepth()([input_layer, nn])
    return keras.layers.Activation(activation)(nn)

def transformer_block(nn, head_num=8, dropout=.1):
    units = nn.shape[-1]
    nn1 = keras.layers.MultiHeadAttention(num_heads=head_num, key_dim=192)(nn, nn)
    nn1 = keras.layers.Dropout(dropout)(nn1)
    nn = tf.add(nn, nn1)
    nn = keras.layers.LayerNormalization()(nn)
    nn1 = keras.layers.Dense(1024)(nn)
    nn1 = keras.layers.Dense(units)(nn1)
    nn1 = keras.layers.Dropout(dropout)(nn1)
    nn = tf.add(nn, nn1)
    return keras.layers.LayerNormalization()(nn)

def stochastic_transformer_block(nn, dropout=.5):
    units = nn.shape[-1]
    x1 = keras.layers.LayerNormalization(epsilon=1e-5)(nn)
    
    attention_output = keras.layers.MultiHeadAttention(num_heads=8, key_dim=192, dropout=.1)(x1, x1)
    
    #attention_output = x1(attention_output)
    
    x2 = tfa.layers.StochasticDepth(dropout)([nn, attention_output])
    #x2 = keras.layers.Add()([attention_output, nn])
    x3 = keras.layers.LayerNormalization(epsilon=1e-5)(x2)
    
    x3 = keras.layers.Dense(units)(x3)
    x3 = keras.layers.Dropout(.1)(x3)

    return tfa.layers.StochasticDepth(dropout)([x2, x3])