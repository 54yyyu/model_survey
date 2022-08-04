from json import load
import h5py
import numpy as np
import os
import wget

#Deepstar dataset
#https://www.dropbox.com/s/0q1jab2dhcg77ld/chip_nexus_binary.h5

def load_data(name='deepstar', input_shape=False, output_shape=False):
    path = './datasets'
    if name=='deepstar':
        x_train, y_train, x_valid, y_valid, x_test, y_test, x_shape, y_shape = load_deepstar()
    
    if input_shape and output_shape:
        return x_train, y_train, x_valid, y_valid, x_test, y_test, x_shape, y_shape
    elif input_shape:
        return x_train, y_train, x_valid, y_valid, x_test, y_test, x_shape
    elif output_shape:
        return x_train, y_train, x_valid, y_valid, x_test, y_test, y_shape
    else:
        return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_deepstar():
    try:
        with h5py.File(os.path.join('.', 'datasets', 'deepstar.h5'), 'r') as hf:
            x_train = np.array(hf['x_train']).astype(np.float32)
            y_train = np.array(hf['y_train']).astype(np.float32).transpose()
            x_valid = np.array(hf['x_valid']).astype(np.float32)
            y_valid = np.array(hf['y_valid']).astype(np.float32).transpose()
            x_test = np.array(hf['x_test']).astype(np.float32)
            y_test = np.array(hf['y_test']).astype(np.float32).transpose()
            
    except FileNotFoundError:
        try:
            os.mkdir('./datasets')
        except FileExistsError:
            pass
        
        wget.download('https://www.dropbox.com/s/g7guotjybwf6p6r/deepstarr_data.h5?dl=1', out=os.path.join('.', 'datasets', 'deepstar.h5'))
        print('\n')
        with h5py.File(os.path.join('.', 'datasets', 'deepstar.h5'), 'r') as hf:
            x_train = np.array(hf['x_train']).astype(np.float32)
            y_train = np.array(hf['y_train']).astype(np.float32).transpose()
            x_valid = np.array(hf['x_valid']).astype(np.float32)
            y_valid = np.array(hf['y_valid']).astype(np.float32).transpose()
            x_test = np.array(hf['x_test']).astype(np.float32)
            y_test = np.array(hf['y_test']).astype(np.float32).transpose()

    return x_train, y_train, x_valid, y_valid, x_test, y_test, (x_train.shape[1], x_train.shape[2]), y_train.shape[1]

"""
if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data('deepstar')
    print(x_train.shape)
"""