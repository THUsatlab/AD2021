#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import test_utility
import os
import json
import pickle
import csv
import librosa
import librosa.display
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from tensorflow.keras.utils import to_categorical

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
import test_utility
from test_utility import DataGenerator

def compute_shape(filename):
  x = np.load(filename)
  shape = x.shape
  return shape
  
def compute_scores(y_true, y_pred, one_hot_y_true, pred_proba):
  accuracy = accuracy_score(y_pred, y_true)
  precision_macro = precision_score(y_pred, y_true, average='macro')
  recall_macro = recall_score(y_pred, y_true, average='macro', labels=np.unique(y_pred))
  f1_score_macro = f1_score(y_pred, y_true, average='macro')
  roc_auc_macro = roc_auc_score(one_hot_y_true, pred_proba, average='macro')

  print(f'Accuracy: {accuracy}')
  print(f'Precision macro: {precision_macro}')
  print(f'Recall macro: {recall_macro}')
  print(f'F1 score macro: {f1_score_macro}')
  print(f'ROC AUC macro: {roc_auc_macro}')

  return accuracy, precision_macro, recall_macro, f1_score_macro, roc_auc_macro

def save_scores(model_name, scores,RESULTS_FOLDER):
  row = [model_name]
  row.extend(scores)
  
  with open(RESULTS_FOLDER + 'genre_all_models_scores.csv', 'a') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(row)   

def txt_label(type,generator,y_pred,RESULTS_FOLDER):
    for i,file in  enumerate(generator.filenames):
        files=file.split('/')
        filename=files[4].split('.')
        filenames=filename[0]
        labels=y_pred[i]+1
        with open(RESULTS_FOLDER +type+'_filename.txt', 'a') as txt_file:
            txt_file.write(filenames)
            txt_file.write('.wav')
            txt_file.write(' ')
            txt_file.write(str(labels))
            txt_file.write('\n')


def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list,  key=lambda x: os.path.getctime(os.path.join(file_path, x)),reverse = True)
        # print(dir_list)
        return dir_list

def load_test_data(DATA_PATH,SPLITS_FOLDER,MELSPECS_FOLDER,MELSPECS_FOLDER_test,SPECS_FOLDER,SPECS_FOLDER_test,MFCCS_FOLDER,MFCCS_FOLDER_test,NUM_CLASSES,NUM_EPOCHS,BATCH_SIZE,MODELS_FOLDER,RESULTS_FOLDER):

    # Mel spectrograms 
    X_train_filenames_melspec = np.load(SPLITS_FOLDER + 'X_train_melspec.npy')
    X_test_filenames_melspec = np.load(SPLITS_FOLDER + 'X_test_melspec.npy')

    one_hot_y_train_melspec = np.load(SPLITS_FOLDER + 'y_train_melspec.npy')

    # Spectrograms
    X_train_filenames_spec = np.load(SPLITS_FOLDER + 'X_train_spec.npy')
    X_test_filenames_spec = np.load(SPLITS_FOLDER + 'X_test_spec.npy')

    one_hot_y_train_spec = np.load(SPLITS_FOLDER + 'y_train_spec.npy')

    # MFCCs
    X_train_filenames_mfcc = np.load(SPLITS_FOLDER + 'X_train_mfcc.npy')
    X_test_filenames_mfcc = np.load(SPLITS_FOLDER + 'X_test_mfcc.npy')

    one_hot_y_train_mfcc = np.load(SPLITS_FOLDER + 'y_train_mfcc.npy')

    print('Mel spectrograms')
    print("Train set:", X_train_filenames_melspec.shape, one_hot_y_train_melspec.shape)
    print("Test set:", X_test_filenames_melspec.shape)

    print('\nSpectrograms')
    print("Train set:", X_train_filenames_spec.shape, one_hot_y_train_spec.shape)
    print("Test set:", X_test_filenames_spec.shape)

    print('\nMFCCs')
    print("Train set:", X_train_filenames_mfcc.shape, one_hot_y_train_mfcc.shape)
    print("Test set:", X_test_filenames_mfcc.shape)

    # Replacing filenames by the absolute path to filenames
    X_train_filenames_melspec = np.array([MELSPECS_FOLDER + fn for fn in X_train_filenames_melspec])
    X_test_filenames_melspec = np.array([MELSPECS_FOLDER_test + fn for fn in X_test_filenames_melspec])

    X_train_filenames_spec = np.array([SPECS_FOLDER + fn for fn in X_train_filenames_spec])
    X_test_filenames_spec = np.array([SPECS_FOLDER_test + fn for fn in X_test_filenames_spec])

    X_train_filenames_mfcc = np.array([MFCCS_FOLDER + fn for fn in X_train_filenames_mfcc])
    X_test_filenames_mfcc = np.array([MFCCS_FOLDER_test + fn for fn in X_test_filenames_mfcc])

    melspec_shape = compute_shape(X_train_filenames_melspec[0])
    spec_shape = compute_shape(X_train_filenames_spec[0])
    mfcc_shape = compute_shape(X_train_filenames_mfcc[0])

    print(f'Mel spectrogram shape: {melspec_shape}')
    print(f'Spectrogram shape: {spec_shape}')
    print(f'MFCC shape: {mfcc_shape}')

    # Mel spectrograms
    test_generator_melspec = DataGenerator(X_test_filenames_melspec, BATCH_SIZE, melspec_shape, 1,shuffle=True)

    # Spectrograms
    test_generator_spec = DataGenerator(X_test_filenames_spec, BATCH_SIZE, spec_shape, 1,shuffle=True)

    # MFCCs
    test_generator_mfcc = DataGenerator(X_test_filenames_mfcc, BATCH_SIZE, mfcc_shape,shuffle=True)
   
    # Loading the best performing model
    mel_models=get_file_list(MODELS_FOLDER+'melspec')
    mel_model=mel_models[0]
    cnn_melspec = load_model(filepath=MODELS_FOLDER + 'melspec/'+mel_model)

    # Loading the best performing model
    spec_models=get_file_list(MODELS_FOLDER+'spec')
    spec_model=spec_models[0]
    cnn_spec = load_model(filepath=MODELS_FOLDER+'spec/'+spec_model)

    # Loading the best performing model
    mfcc_models=get_file_list(MODELS_FOLDER+'mfcc')
    mfcc_model=mfcc_models[0]
    cnn_mfcc = load_model(filepath=MODELS_FOLDER+'mfcc/'+mfcc_model)

    #-=========melspec=============
    # Making predictions on test set
    TEST_STEPS_MELSPEC = np.ceil(len(X_test_filenames_melspec)/BATCH_SIZE)
    pred_proba_melspec = cnn_melspec.predict(x=test_generator_melspec, steps=TEST_STEPS_MELSPEC)

    y_pred_melspec = np.argmax(pred_proba_melspec, axis=1)
    
    txt_label('melspec',test_generator_melspec,y_pred_melspec,RESULTS_FOLDER)

    #-=========Spectrograms=============
    # Making predictions on test set
    TEST_STEPS_SPEC = np.ceil(len(X_test_filenames_spec)/BATCH_SIZE)
    pred_proba_spec = cnn_spec.predict(x=test_generator_spec, steps=TEST_STEPS_SPEC)

    y_pred_spec = np.argmax(pred_proba_spec, axis=1)
    
    txt_label('spec',test_generator_spec,y_pred_spec,RESULTS_FOLDER)
    #==========mfcc================
    # Making predictions on test set
    TEST_STEPS_MFCC = np.ceil(len(X_test_filenames_mfcc)/BATCH_SIZE)
    pred_proba_mfcc = cnn_mfcc.predict(x=test_generator_mfcc, steps=TEST_STEPS_MFCC)
    y_pred_mfcc = np.argmax(pred_proba_mfcc, axis=1)
   
    txt_label('mfcc',test_generator_mfcc,y_pred_mfcc,RESULTS_FOLDER)

