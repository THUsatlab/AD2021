#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, Dataset

def compute_shape(filename):
    x = np.load(filename)
    shape = x.shape
    return shape

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
        X_train, X_val, X_test, y_train_labels, y_val_labels, y_test {ndarray} -- train, validation, and test sets
    """
    assert train_size >= 0 and train_size <= 1, "Invalid training set fraction"

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    splits = splitter.split(X, y)

    for train_index, val_index in splits:
        X_train, X_val = X[train_index], X[val_index]
        y_train_labels, y_val_labels = y[train_index], y[val_index]
    return X_train, X_val,  y_train_labels, y_val_labels

class TensorTrainDataset(Dataset):
    def __init__(self, filenames, labels, input_shape):
        self.filenames = filenames
        self.labels = labels
        self.input_shape = input_shape

    def __getitem__(self, index):
        filename = self.filenames[index]
        data = np.load(str(filename))
        
        label = self.labels[index]
        return torch.from_numpy(data), torch.tensor(label, dtype=torch.int64)

    def __len__(self):
        return len(self.filenames)

def load_train_dataset(dataset_path, batch_size):
    label_names = ['AD', 'MCI', 'HC']

    filename_list = []
    label_list = []
    for idx, label_name in enumerate(label_names):
        if os.path.isdir(os.path.join(dataset_path, label_name)):
            fns = os.listdir(os.path.join(dataset_path, label_name))
            for fn in fns:
                # Replacing filenames by the absolute path to filenames
                filename_list.append(os.path.join(dataset_path, label_name, fn))
                label_list.append(idx)

    filename_list = np.array(filename_list)
    label_list = np.array(label_list)
    
    feature_shape = compute_shape(filename_list[0])

    # Splitting the filenames into train, validation, and test sets with respect to the relative distribution of instances per category in our dataset
    X_train_filenames, X_val_filenames, y_train_labels, y_val_labels = data_split(filename_list, label_list, train_size=0.8)

    train_dataset = TensorTrainDataset(X_train_filenames, y_train_labels, feature_shape)
    val_dataset = TensorTrainDataset(X_val_filenames, y_val_labels, feature_shape)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, feature_shape

class TensorTestDataset(Dataset):
    def __init__(self, filenames, input_shape):
        self.filenames = filenames
        self.input_shape = input_shape

    def __getitem__(self, index):
        filename = self.filenames[index]
        data = np.load(str(filename))
        
        return torch.from_numpy(data), os.path.basename(filename)

    def __len__(self):
        return len(self.filenames)

def load_test_dataset(dataset_path, batch_size):
    filename_list = []
    fns = os.listdir(dataset_path)
    for fn in fns:
        if os.path.splitext(fn)[-1] == '.npy':
            # Replacing filenames by the absolute path to filenames
            filename_list.append(os.path.join(dataset_path, fn))

    filename_list = np.array(filename_list)
    
    feature_shape = compute_shape(filename_list[0])

    test_dataset = TensorTestDataset(filename_list, feature_shape)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, feature_shape
