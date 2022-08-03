from model_zoo import deepstar
from helper import *
import tensorflow as tf
from tensorflow import keras
from scipy.stats import spearmanr

def spearman_r(y_true, y_pred):
    return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), 
            tf.cast(y_true, tf.float32)], Tout = tf.float32) )

def create_deepstar(input_shape):
    inputs, outputs = deepstar.create(input_shape)
    return keras.Model(inputs=inputs, outputs=outputs)

def train_deepstar():
    x_train, y_train, x_valid, y_valid, x_test, y_test, x_shape, y_shape = load_deepstar()

    model = create_deepstar(x_shape)
    tasks = ['Dev','Hk']
    
    metrics = [spearman_r]
    
    model.compile(keras.optimizers.Adam(lr=0.002),
                    loss='mse',
                    metrics=metrics)
    
    print(model.summary())
    
    # early stopping callback
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', #'val_aupr',#
                                                patience=10, 
                                                verbose=1, 
                                                mode='min', 
                                                restore_best_weights=True)
    # reduce learning rate callback
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                    factor=0.2,
                                                    patience=3, 
                                                    min_lr=1e-7,
                                                    mode='min',
                                                    verbose=1) 

    # train model
    history = model.fit(x_train, y_train, 
                        epochs=100,
                        batch_size=128, 
                        shuffle=True,
                        validation_data=(x_valid, y_valid), 
                        callbacks=[es_callback, reduce_lr])
    
    results = model.evaluate(x_test, y_test, verbose=1)
    #print(results)
    