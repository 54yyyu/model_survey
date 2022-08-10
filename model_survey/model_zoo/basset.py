import tensorflow as tf
from tensorflow import keras

def create(input_shape=(100, 4)):
    # input layer
    L, A = input_shape
    inputs = keras.layers.Input(shape=(L,A))
    # layer 1 - convolution
    nn = keras.layers.Conv1D(filters=300, kernel_size=19, padding=‘same’)(inputs)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(‘relu’)(nn)
    nn = keras.layers.MaxPool1D(pool_size=3)(nn)
    # layer 2 - convolution
    nn = keras.layers.Conv1D(filters=200, kernel_size=11, padding=‘same’)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(‘relu’)(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    # layer 3 - convolution
    nn = keras.layers.Conv1D(filters=200, kernel_size=7, padding=‘same’)(nn)
    nn = keras.layers.BatchNormalization()(nn)
    nn = keras.layers.Activation(‘relu’)(nn)
    nn = keras.layers.MaxPool1D(pool_size=4)(nn)
    # layer 3 - Fully-connected
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(1000)(nn)
    nn = keras.layers.Activation(‘relu’)(nn)
    nn = keras.layers.Dropout(0.3)(nn)
    # layer 4 - Fully-connected
    nn = keras.layers.Flatten()(nn)
    nn = keras.layers.Dense(1000)(nn)
    nn = keras.layers.Activation(‘relu’)(nn)
    nn = keras.layers.Dropout(0.3)(nn)
    # Output layer
    logits = keras.layers.Dense(output_shape, use_bias=True)(nn)
    outputs = keras.layers.Activation(‘sigmoid’)(logits)
    # create keras model
    model = keras.Model(inputs=inputs, outputs=outputs)