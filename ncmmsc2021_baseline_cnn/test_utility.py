import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

class DataGenerator(tf.keras.utils.Sequence):
   
    def __init__(self, filenames, batch_size, input_shape, n_channels=None,shuffle=True):
        self.filenames = filenames
        #self.labels = labels
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_channels = n_channels
        self.shuffle = shuffle

    def __len__(self):
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):

        # Loading batches of images
        batch_X_filenames = self.filenames[idx * self.batch_size : (idx+1) * self.batch_size]

        if self.n_channels:
            inpt_shape = (self.input_shape[0], self.input_shape[1], self.n_channels)
            batch_X_arr = [np.load(str(fn)).reshape(inpt_shape) for fn in batch_X_filenames]
        else:
            batch_X_arr = [np.load(str(fn)) for fn in batch_X_filenames]
        
        batch_X = np.array(batch_X_arr)

        return batch_X



def load_data(dirname):

    X_filenames = []
    for fn in os.listdir(dirname):
        X_filenames.append(fn)

    # Converting the lists in numpy arrays
    X_filenames = np.array(X_filenames)
    print("Loading data done!")

    return X_filenames
