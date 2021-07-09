import sys
import os
import json
import pickle
import csv
import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Activation, BatchNormalization, ReLU, Conv1D, MaxPooling1D
from tensorflow.keras import utils
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import utility
from utility import DataGenerator

DRIVE_PATH = '../data/'
sys.path.append(DRIVE_PATH)

NUM_CLASSES = 3
NUM_EPOCHS = 30
BATCH_SIZE = 128
def compute_shape(filename):
    x = np.load(filename)
    shape = x.shape
    return shape
def build_cnn_2d(input_shape, nb_filters, dense_units, output_shape=NUM_CLASSES, activation='softmax', dropout=0.3, poolings=None):
    
    n_mels = input_shape[0]
    
    if not poolings:
      if n_mels >= 256:
          poolings = [(2, 4), (4, 4), (4, 5), (2, 4), (4, 4)]
      elif n_mels >= 128:
          poolings = [(2, 4), (4, 4), (2, 5), (2, 4), (4, 4)]
      elif n_mels >= 96:
          poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (4, 4)]
      elif n_mels >= 72:
          poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (3, 4)]
      elif n_mels >= 64:
          poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (4, 4)]
    
    # Input block
    melspec_input = Input(shape=input_shape, name='input')

    # Conv block 1
    x = Conv2D(nb_filters[0], (3, 3), activation='relu', name='conv_1')(melspec_input)
    x = MaxPooling2D(pool_size=poolings[0], strides=2, padding='same', name='pool_1')(x)
    x = BatchNormalization(name='bn_1')(x)
    
    # Conv block 2
    x = Conv2D(nb_filters[1], (3, 3), activation='relu', name='conv_2')(x)
    x = MaxPooling2D(pool_size=poolings[1], strides=2, padding='same', name='pool_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    
    # Conv block 3
    x = Conv2D(nb_filters[2], (3, 3), activation='relu', name='conv_3')(x)
    x = MaxPooling2D(pool_size=poolings[2], strides=2, padding='same', name='pool_3')(x)
    x = BatchNormalization(name='bn_3')(x)
    
    # Flattening the output and feeding it into dense layer
    x = Flatten(name='flatten')(x)
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(0.001), name='dense')(x)
    x = Dropout(dropout, name='dropout')(x)
    
    # Output Layer
    x = Dense(output_shape, activation=activation, kernel_regularizer=l2(0.001), name = 'dense_output')(x)
    
    # Create model
    model = Model(melspec_input, x)
    
    return model
    
def build_cnn_1d(input_shape, nb_filters, dense_units, output_shape=NUM_CLASSES, activation='softmax', dropout=0.3, poolings=None):
    
  model = Sequential()

  # Conv block 1
  model.add(Conv1D(nb_filters[0], 3, activation='relu', input_shape=input_shape, name='conv_1'))
  model.add(MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool_1'))
  model.add(BatchNormalization(name='bn_1'))
  
  # Conv block 2
  model.add(Conv1D(nb_filters[1], 3, activation='relu', name='conv_2'))
  model.add(MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool_2'))
  model.add(BatchNormalization(name='bn_2'))
  
  # Conv block 3
  model.add(Conv1D(nb_filters[2], 3, activation='relu', name='conv_3'))
  model.add(MaxPooling1D(pool_size=3, strides=2, padding='same', name='pool_3'))
  model.add(BatchNormalization(name='bn_3'))
  
  # Flattening the output and feeding it into dense layer
  model.add(Flatten(name='flatten'))
  model.add(Dense(dense_units, activation='relu',kernel_regularizer=l2(0.001),name='dense'))
  model.add(Dropout(dropout, name='dropout'))
  
  # Output Layer
  model.add(Dense(output_shape, activation=activation,kernel_regularizer=l2(0.001), name = 'dense_output'))
  
  return model
  
def create_checkpoint(filepath):
  checkpoint = ModelCheckpoint(filepath, 
                              monitor='val_acc', 
                              verbose=0, 
                              save_best_only=True,
                              save_weights_only=False, 
                              mode='auto', 
                              period=1)
  return checkpoint

reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', 
                                         factor=0.1, 
                                         patience=1, 
                                         verbose=0)
                                         
def train_model(DATA_PATH,SPLITS_FOLDER,MELSPECS_FOLDER,SPECS_FOLDER,MFCCS_FOLDER,MODELS_FOLDER,RESULTS_FOLDER,PICKLES_FOLDER,NUM_CLASSES,NUM_EPOCHS,BATCH_SIZE):

# Loading the label map
    with open(DATA_PATH + 'genre_label_map.json', 'r') as output:
        label_map = json.load(output)

    print(label_map)

    # Mel spectrograms 
    X_train_filenames_melspec = np.load(SPLITS_FOLDER + 'X_train_melspec.npy')
    one_hot_y_train_melspec = np.load(SPLITS_FOLDER + 'y_train_melspec.npy')
    # Spectrograms
    X_train_filenames_spec = np.load(SPLITS_FOLDER + 'X_train_spec.npy')
    one_hot_y_train_spec = np.load(SPLITS_FOLDER + 'y_train_spec.npy')
    # MFCCs
    X_train_filenames_mfcc = np.load(SPLITS_FOLDER + 'X_train_mfcc.npy')
    one_hot_y_train_mfcc = np.load(SPLITS_FOLDER + 'y_train_mfcc.npy')
    print('Mel spectrograms')
    print("Train set:", X_train_filenames_melspec.shape, one_hot_y_train_melspec.shape)
    print('\nSpectrograms')
    print("Train set:", X_train_filenames_spec.shape, one_hot_y_train_spec.shape)
    print('\nMFCCs')
    print("Train set:", X_train_filenames_mfcc.shape, one_hot_y_train_mfcc.shape)
    # Replacing filenames by the absolute path to filenames
    X_train_filenames_melspec = np.array([MELSPECS_FOLDER + fn for fn in X_train_filenames_melspec])

    X_train_filenames_spec = np.array([SPECS_FOLDER + fn for fn in X_train_filenames_spec])

    X_train_filenames_mfcc = np.array([MFCCS_FOLDER + fn for fn in X_train_filenames_mfcc])
    melspec_shape = compute_shape(X_train_filenames_melspec[0])
    spec_shape = compute_shape(X_train_filenames_spec[0])
    mfcc_shape = compute_shape(X_train_filenames_mfcc[0])

    # Splitting the filenames into train, validation, and test sets with respect to the relative distribution of instances per category in our dataset
    X_train_filenames_melspec1, X_test_filenames_melspec1, one_hot_y_train_melspec1, one_hot_y_test_melspec1 = utility.data_split(X_train_filenames_melspec, one_hot_y_train_melspec, train_size=0.8)

    X_train_filenames_spec1, X_test_filenames_spec1, one_hot_y_train_spec1, one_hot_y_test_spec1= utility.data_split(X_train_filenames_spec, one_hot_y_train_spec, train_size=0.8)

    X_train_filenames_mfcc1, X_test_filenames_mfcc1, one_hot_y_train_mfcc1, one_hot_y_test_mfcc1= utility.data_split(X_train_filenames_mfcc, one_hot_y_train_mfcc, train_size=0.8)

    # Mel spectrograms
    train_generator_melspec = DataGenerator(X_train_filenames_melspec1, one_hot_y_train_melspec1, BATCH_SIZE, melspec_shape, 1,shuffle=True)
    test_generator_melspec = DataGenerator(X_test_filenames_melspec1, one_hot_y_test_melspec1, BATCH_SIZE, melspec_shape, 1,shuffle=True)

    # Spectrograms
    train_generator_spec = DataGenerator(X_train_filenames_spec1, one_hot_y_train_spec1, BATCH_SIZE, spec_shape, 1,shuffle=True)
    test_generator_spec = DataGenerator(X_test_filenames_spec1, one_hot_y_test_spec1, BATCH_SIZE, spec_shape, 1,shuffle=True)

    # MFCCs
    train_generator_mfcc = DataGenerator(X_train_filenames_mfcc1, one_hot_y_train_mfcc1, BATCH_SIZE, mfcc_shape,shuffle=True)
    test_generator_mfcc = DataGenerator(X_test_filenames_mfcc1, one_hot_y_test_mfcc1, BATCH_SIZE, mfcc_shape,shuffle=True)
    #=============3.1. CNN for mel Spectrograms==========================
    cnn_melspec = build_cnn_2d((melspec_shape[0],melspec_shape[1], 1), [32,32,32], 64)
    cnn_melspec.summary()
    # Compiling our neural network
    cnn_melspec.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(0.001),
                  metrics=['accuracy'])

    # Creating callbacks list
    filepath = MODELS_FOLDER + 'melspec/cnn_3_mel_epoch_{epoch:02d}_acc_{acc:.4f}_val_acc_{val_acc:.4f}.h5'

    callbacks_list = [create_checkpoint(filepath), reduce_lr_on_plateau]

    STEPS_PER_EPOCH = np.ceil(len(X_train_filenames_melspec1)/BATCH_SIZE)
    VAL_STEPS = np.ceil(len(X_test_filenames_melspec1)/BATCH_SIZE)

    cnn_melspec_hist = cnn_melspec.fit(x=train_generator_melspec,
                                      epochs=NUM_EPOCHS,
                                      steps_per_epoch=STEPS_PER_EPOCH,
                                      validation_data=test_generator_melspec,
                                      validation_steps=VAL_STEPS,
                                      shuffle=True,
                                      callbacks=callbacks_list)

    # Saving scores on train and validation sets
    with open(PICKLES_FOLDER + 'exp_1_cnn_3_mel_history.pkl', 'wb') as f:
        pickle.dump(cnn_melspec_hist.history, f)

    # Loading scores
    with open(PICKLES_FOLDER + 'exp_1_cnn_3_mel_history.pkl', 'rb') as f:
        cnn_melspec_hist_dict = pickle.load(f)

    #=============3.2. CNN for Spectrograms==========================
    cnn_spec = build_cnn_2d((spec_shape[0], spec_shape[1], 1), [32,32,32], 64)
    cnn_spec.summary()

    # Compiling our neural network
    cnn_spec.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(0.001),
                  metrics=['accuracy'])

    # Creating callbacks list
    filepath = MODELS_FOLDER + 'spec/cnn_3_spec_epoch_{epoch:02d}_acc_{acc:.4f}_val_acc_{val_acc:.4f}.h5'

    callbacks_list = [create_checkpoint(filepath), reduce_lr_on_plateau]

    STEPS_PER_EPOCH = np.ceil(len(X_train_filenames_spec1)/BATCH_SIZE)
    VAL_STEPS = np.ceil(len(X_test_filenames_spec1)/BATCH_SIZE)

    # 3H15
    cnn_spec_hist = cnn_spec.fit(x=train_generator_spec,
                                  epochs=NUM_EPOCHS,
                                  steps_per_epoch=STEPS_PER_EPOCH,
                                  validation_data=test_generator_spec,
                                  validation_steps=VAL_STEPS,
                                  shuffle=True,
                                  callbacks=callbacks_list)
    # Saving scores on train and validation sets
    with open(PICKLES_FOLDER + 'exp_1_cnn_3_spec_history.pkl', 'wb') as f:
        pickle.dump(cnn_spec_hist.history, f)

    # Loading scores
    with open(PICKLES_FOLDER + 'exp_1_cnn_3_spec_history.pkl', 'rb') as f:
        cnn_spec_hist_dict = pickle.load(f)

    ####==================3.3. CNN for MFCC======================
    cnn_mfcc = build_cnn_1d(mfcc_shape, [32,32,32], 64)
    cnn_mfcc.summary()

    # Compiling our neural network
    cnn_mfcc.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(0.001),
                  metrics=['accuracy'])

    # Creating callbacks list
    filepath = MODELS_FOLDER + 'mfcc/cnn_3_mfcc_epoch_{epoch:02d}_acc_{acc:.4f}_val_acc_{val_acc:.4f}.h5'

    callbacks_list = [create_checkpoint(filepath), reduce_lr_on_plateau]

    STEPS_PER_EPOCH = np.ceil(len(X_train_filenames_mfcc1)/BATCH_SIZE)
    VAL_STEPS = np.ceil(len(X_test_filenames_mfcc1)/BATCH_SIZE)

    cnn_mfcc_hist = cnn_mfcc.fit(x=train_generator_mfcc,
                                epochs=NUM_EPOCHS,
                                steps_per_epoch=STEPS_PER_EPOCH,
                                validation_data=test_generator_mfcc,
                                validation_steps=VAL_STEPS,
                                shuffle=True,
                                callbacks=callbacks_list)

    # Saving scores on train and validation sets
    with open(PICKLES_FOLDER + 'exp_1_cnn_3_mfcc_history.pkl', 'wb') as f:
        pickle.dump(cnn_mfcc_hist.history, f)

    # Loading scores
    with open(PICKLES_FOLDER + 'exp_1_cnn_3_mfcc_history.pkl', 'rb') as f:
        cnn_mfcc_hist_dict = pickle.load(f)