import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

class DataGenerator(tf.keras.utils.Sequence):
    """Generate batches of tensor image data and their corresponding labels.
    
        Arguments:
            filenames {ndarray} -- array of filenames
            labels {ndarray} -- array containing one-hot encoded or categorical labels for each track
            batch_size {int} -- size of the batches
            input_shape {tuple of int} -- (image width, image height) 
            n_channels {int} (default: {None})-- number of channels (1 for black/white, 3 for RGB)
    """
    
    def __init__(self, filenames, labels, batch_size, input_shape, n_channels=None,shuffle=True):
        self.filenames = filenames
        self.labels = labels
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

        # Loading batches of corresponding labels
        batch_y = np.array(self.labels[idx * self.batch_size : (idx+1) * self.batch_size])

        return batch_X, batch_y



def load_data(dirname, label_map):
    """Creates an array of the filenames in `dirname` and another for the corresponding labels.

    Arguments:
        dirname {str} -- directory path 
        label_map {dict} -- dictionary mapping a genre to a numeric label

    Returns:
        (ndarray, ndarray) -- arrays of filenames and labels
    """

    # Creating empty lists for the filenames and corresponding labels with respect to a label map
    X_filenames, y = [], []

    # Looping through each file
    for fn in os.listdir(dirname):

            # Extracting the semantic label
          #  filename_components = fn.split(".")
            filename_components = fn.split("_") 
            semantic_label = filename_components[0]
            
            # Saving the absolute path to the filename and its corresponding label
            X_filenames.append(fn)
            y.append(label_map[semantic_label])

    # Converting the lists in numpy arrays
    X_filenames = np.array(X_filenames)
    y = np.array(y)
    print("Loading data done!")

    return X_filenames, y

def data_split(X, y, train_size, random_state=42):
    """Randomly split dataset with respect to the relative distribution of instances per category in our dataset,
    based on these ratios:

        'train': train_size
        'validation': (1-train_size) / 2
        'test':  (1-train_size) / 2

    Eg: passing train_size=0.8 gives a 80% / 10% / 10% split

    Arguments:
        X {ndarray} -- samples to be split
        y {ndarray} -- labels
        train_size {float or int} -- ratio of train set to whole dataset 
        random_state {int or RandomState instance,} -- controls the randomness of the training and testing indices produced (default: {None})

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test {ndarray} -- train, validation, and test sets
    """
    assert train_size >= 0 and train_size <= 1, "Invalid training set fraction"

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    splits = splitter.split(X, y)

    for train_index, val_index in splits:
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
    return X_train, X_val,  y_train, y_val

