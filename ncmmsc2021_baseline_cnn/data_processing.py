#!/usr/bin/python
# -*- coding: UTF-8 -*-
import utility
import os
import math
import json
import numpy as np
import librosa
import librosa.display

import matplotlib as mpl
mpl.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from tensorflow.keras.utils import to_categorical

#检测数据大小是否一致
#======================checksize ==================================
def check_size(dataset_path, compute_spec=False, compute_melspec=False, 
               n_fft=1024, hop_length=512, segment_duration=3, segment_overlap=0.5, num_segments=None,SAMPLE_RATE = 22050,DURATION = 6,sr=22050,
               n_mels=128):
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    #创建一个空list
    if compute_spec:
        sizes_spec = []
    if compute_melspec:
        sizes_melspec = []
    #计算分割段
    if num_segments:
        num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    else:
        num_samples_per_segment = int(SAMPLE_RATE * segment_duration)
        num_segments = int((DURATION / (segment_duration * segment_overlap)) - 1)
        
    for dirpath, _, filenames in os.walk(dataset_path):

        if dirpath is not dataset_path:

            for fn in filenames:
                print('Check:  ',fn)
                
                file_path = os.path.join(dirpath, fn)
                y, _ = librosa.load(file_path, sr=SAMPLE_RATE)

                for s in range(num_segments):
                    start_sample = int(num_samples_per_segment * segment_overlap * s) 
                    finish_sample = start_sample + num_samples_per_segment
                    
                    ##### SPECS 语谱图 #####
                    if compute_spec:
                        spec = librosa.core.stft(y[start_sample:finish_sample],
                                                        n_fft=n_fft,
                                                        hop_length=hop_length)

                        spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max) # Converting to decibals
                        
                        sizes_spec.append(spec.shape)
                    
                    ##### MEL SPECS mel谱图 #####
                    if compute_melspec:
                        
                        melspec = librosa.feature.melspectrogram(y=y[start_sample:finish_sample], 
                                                                 sr=sr, 
                                                                 n_fft=n_fft, 
                                                                 hop_length=hop_length,
                                                                 n_mels=n_mels)

                        melspec = librosa.power_to_db(melspec, ref=np.max)
                        
                        sizes_melspec.append(melspec.shape)
                        

    # Checking if all spectrograms/mel spectrograms are the same size
    if compute_spec:
        same_size_spec = (len(set(sizes_spec)) == 1)
        max_size_spec = max(sizes_spec)
        print(f'Do all the spectrograms have the same size? {same_size_spec}')
        print(f'The maximum size of spectrograms: {max_size_spec}')
        
    if compute_melspec:
        same_size_melspec = (len(set(sizes_melspec)) == 1)
        max_size_melspec = max(sizes_melspec)
        print(f'Do all the mel spectrograms have the same size? {same_size_melspec}')
        print(f'The maximum size of mel spectrograms: {max_size_melspec}')


def create_dataset(dataset_path, melspecs_folder=None, specs_folder=None, mfccs_folder=None,
                   n_fft=1024, hop_length=512, segment_duration=3, segment_overlap=0.5, SAMPLE_RATE = 22050,DURATION = 6,sr=22050,
                   num_segments=None, n_mels=128, n_mfcc=20):

    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    # Computing the number of samples per segment 
    if num_segments:
        num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    else:
        num_samples_per_segment = int(SAMPLE_RATE * segment_duration)
        num_segments = int((DURATION / (segment_duration * segment_overlap)) - 1)

    # Computing the expected number of mfcc vectors per segment, or number of frames per segment
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) 

    # Looping through all the genres
    for dirpath, _, filenames in os.walk(dataset_path):

        if dirpath is not dataset_path:

            dirpath_components = dirpath.split("/") # data/genres/blues => ["data", "genres", "blues"]
            semantic_label = dirpath_components[-1]
            print("Processing {}".format(semantic_label))
            
            for fn in filenames:
                
                file_path = os.path.join(dirpath, fn)
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                for s in range(num_segments):
                    start_sample = int(num_samples_per_segment * segment_overlap * s)
                    finish_sample = start_sample + num_samples_per_segment
                    
                    ##### Start: MEL SPECTROGRAMS #####
                    if melspecs_folder:
                        melspec = librosa.feature.melspectrogram(y=y[start_sample:finish_sample], 
                                                                sr=sr, 
                                                                n_fft=n_fft, 
                                                                hop_length=hop_length,
                                                                n_mels=n_mels)
                        
                        melspec = librosa.power_to_db(melspec, ref=np.max)
                        
                        # Saving the mel spectrogram into a .npy file
                        npy_filename = melspecs_folder + fn.replace('.wav', f'.{s}.melspec.npy')
                        np.save(npy_filename, melspec)
                    ##### End: MEL SPECTROGRAMS #####

                    ##### Start: SPECTROGRAMS #####
                    if specs_folder:
                        spec = librosa.core.stft(y[start_sample:finish_sample],
                                                n_fft=n_fft,
                                                hop_length=hop_length)
                        
                        spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max) 

                        # Saving the spectrogram into a .npy file
                        npy_filename = specs_folder + fn.replace('.wav', f'.{s}.spec.npy')
                        np.save(npy_filename, spec)
                    ##### End: SPECTROGRAMS #####

                    ##### Start: MFCCs #####
                    if mfccs_folder:
                        mfcc = librosa.feature.mfcc(y[start_sample:finish_sample],
                                                    sr=sr,
                                                    n_fft=n_fft,
                                                    n_mfcc=n_mfcc,
                                                    hop_length=hop_length)
                     
                        if mfcc.shape[1] == expected_num_mfcc_vectors_per_segment:
                            npy_filename = mfccs_folder + fn.replace('.wav', f'.{s}.mfcc.npy')
                            np.save(npy_filename, mfcc)
                    ##### End: MFCCs #####
            
    print('Creating dataset done!')



def save_feature_train(DATA_PATH,MELSPECS_FOLDER,SPECS_FOLDER,MFCCS_FOLDER,SPLITS_FOLDER):

# Loading the label map
    with open(DATA_PATH + 'genre_label_map.json', 'r') as output:
        label_map = json.load(output)

    X_filenames_melspec, y_melspec = utility.load_data(MELSPECS_FOLDER, label_map)
    X_filenames_spec, y_spec = utility.load_data(SPECS_FOLDER, label_map)
    X_filenames_mfcc, y_mfcc = utility.load_data(MFCCS_FOLDER, label_map)

    # One hot encoding the labels one-hot编码
    one_hot_y_melspec = to_categorical(y_melspec, num_classes=3)
    one_hot_y_spec = to_categorical(y_spec, num_classes=3)
    one_hot_y_mfcc = to_categorical(y_mfcc, num_classes=3)

    X_train_filenames_melspec=X_filenames_melspec
    X_train_filenames_spec=X_filenames_spec
    X_train_filenames_mfcc=X_filenames_mfcc
    one_hot_y_train_melspec =one_hot_y_melspec
    one_hot_y_train_spec=one_hot_y_spec
    one_hot_y_train_mfcc=one_hot_y_mfcc

    np.save(SPLITS_FOLDER + 'X_train_melspec.npy', X_train_filenames_melspec)
    
    np.save(SPLITS_FOLDER + 'X_train_spec.npy', X_train_filenames_spec)

    np.save(SPLITS_FOLDER + 'X_train_mfcc.npy', X_train_filenames_mfcc)

    np.save(SPLITS_FOLDER + 'y_train_melspec.npy', one_hot_y_train_melspec)

    np.save(SPLITS_FOLDER + 'y_train_spec.npy', one_hot_y_train_spec)

    np.save(SPLITS_FOLDER + 'y_train_mfcc.npy', one_hot_y_train_mfcc)
