import numpy as np

from . import model_zoo
from . import helper
from re import X
import tensorflow as tf
from tensorflow import keras
from scipy.stats import spearmanr, pearsonr

def spearman_r(y_true, y_pred):
    return (tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), 
            tf.cast(y_true, tf.float32)], Tout = tf.float32) )

def pearson_r(y_true, y_pred):
    return (tf.py_function(pearsonr, [tf.cast(y_pred, tf.float32), 
            tf.cast(y_true, tf.float32)], Tout = tf.float32) )

def create_deepstar(input_shape):
    inputs, outputs = model_zoo.deepstar.create(input_shape)
    return keras.Model(inputs=inputs, outputs=outputs)

def create_deepstar(input_shape):
    inputs, outputs = model_zoo.deepstar.create(input_shape)
    return keras.Model(inputs=inputs, outputs=outputs)

def train_deepstar():
    x_train, y_train, x_valid, y_valid, x_test, y_test, x_shape, y_shape = helper.load_deepstar()
    
    model = create_deepstar(x_shape)
    tasks = ['Dev','Hk']
    
    metrics = [pearson_r, spearman_r]
    
    model.compile(keras.optimizers.Adam(learning_rate=0.002),
                    loss='mse',
                    metrics=metrics)
    
    print(model.summary())
    
    pred = np.array(model.predict(x_test))
    pred_a = pred[:, 0]
    pred_b = pred[:, 1]
    
    print('pearsonr 0:', pearsonr(pred_a.flatten(), y_test[:, 0].flatten())[0])
    print('pearsonr 1:', pearsonr(pred_b.flatten(), y_test[:, 1].flatten())[0])
    
    
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
                        epochs=50,
                        batch_size=128, 
                        shuffle=True,
                        validation_data=(x_valid, y_valid), 
                        callbacks=[es_callback, reduce_lr])
    
    results = model.evaluate(x_test, y_test, verbose=1)

    pred = np.array(model.predict(x_test))
    pred_a = pred[:, 0]
    pred_b = pred[:, 1]
    
    print('pearsonr 0:', pearsonr(pred_a.flatten(), y_test[:, 0].flatten())[0])
    print('pearsonr 1:', pearsonr(pred_b.flatten(), y_test[:, 1].flatten())[0])

    #print(results)
    
class EvaluateCallback(keras.callbacks.Callback):
    def __init__(self, x_valid, y_valid, x_test, y_test, bs):
        super(EvaluateCallback, self).__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.batch_size = bs
    
    def on_epoch_end(self, epoch, logs=None):

        pred = np.array(self.model.predict(self.x_test, batch_size=self.batch_size))
        pred_a = pred[:, 0]
        pred_b = pred[:, 1]

        logs = logs or {}
        logs['test_pearson_0'] = pearsonr(pred_a.flatten(), self.y_test[:, 0].flatten())[0]
        logs['test_pearson_1'] = pearsonr(pred_b.flatten(), self.y_test[:, 1].flatten())[0]
        
        #logs['val_aupr'] = keras.metrics.AUC(curve='pr', name='aupr')()
        
        
