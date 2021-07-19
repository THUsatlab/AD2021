#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import argparse
import math
import numpy as np
import librosa
import librosa.display

# Check if the data size is the same
#======================checksize ==================================
def check_size(dataset_path, compute_spec=False, compute_melspec=False, 
               n_fft=1024, hop_length=512, segment_duration=3, 
               segment_overlap=0.5, num_segments=None, 
               SAMPLE_RATE=22050, DURATION=6, sr=22050,
               n_mels=128):
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    # Create an empty list
    if compute_spec:
        sizes_spec = []
    if compute_melspec:
        sizes_melspec = []
    # Compute segment
    if num_segments:
        num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    else:
        num_samples_per_segment = int(SAMPLE_RATE * segment_duration)
        num_segments = int((DURATION / (segment_duration * segment_overlap)) - 1)
        
    for dirpath, _, filenames in os.walk(dataset_path):

        if dirpath is not dataset_path:

            for fn in filenames:
                if os.path.splitext(fn)[-1] != '.wav':
                    continue

                print('Check:  ',fn)
                
                file_path = os.path.join(dirpath, fn)
                y, _ = librosa.load(file_path, sr=SAMPLE_RATE)

                for s in range(num_segments):
                    start_sample = int(num_samples_per_segment * segment_overlap * s) 
                    finish_sample = start_sample + num_samples_per_segment
                    
                    ##### SPECS spectrogram #####
                    if compute_spec:
                        spec = librosa.core.stft(y[start_sample:finish_sample],
                                                        n_fft=n_fft,
                                                        hop_length=hop_length)

                        spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max) # Converting to decibals
                        
                        sizes_spec.append(spec.shape)
                    
                    ##### MEL SPECS mel spectrogram #####
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


#检测数据大小是否一致
#======================checksize ==================================
def check_size_for_test(dataset_path, compute_spec=False, compute_melspec=False, 
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

        if dirpath is dataset_path:

            for fn in filenames:
                if os.path.splitext(fn)[-1] != '.wav':
                    continue

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
                        

    # 检测谱图大小是否一致
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


def create_dataset(audio_path, melspecs_folder=None, specs_folder=None, mfccs_folder=None,
                   n_fft=1024, hop_length=512, segment_duration=3, segment_overlap=0.5, 
                   SAMPLE_RATE=22050, DURATION=6, sr=22050,
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
    for dirpath, _, filenames in os.walk(audio_path):

        if dirpath is not audio_path:
            dirpath_components = os.path.split(dirpath) # data/genres/blues => ["data", "genres", "blues"]
            semantic_label = dirpath_components[-1]
            print("Processing {}".format(semantic_label))
            
            for fn in filenames:
                if os.path.splitext(fn)[-1] != '.wav':
                    continue

                print('Processing:  ',fn)

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
                        feature_save_dir = os.path.join(melspecs_folder, semantic_label)
                        if not os.path.isdir(feature_save_dir):
                            os.makedirs(feature_save_dir)
                        npy_filename = os.path.join(feature_save_dir, fn.replace('.wav', f'.{s}.melspec.npy'))
                        np.save(npy_filename, melspec)
                    ##### End: MEL SPECTROGRAMS #####

                    ##### Start: SPECTROGRAMS #####
                    if specs_folder:
                        spec = librosa.core.stft(y[start_sample:finish_sample],
                                                n_fft=n_fft,
                                                hop_length=hop_length)
                        
                        spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max) 

                        # Saving the spectrogram into a .npy file
                        feature_save_dir = os.path.join(specs_folder, semantic_label)
                        if not os.path.isdir(feature_save_dir):
                            os.makedirs(feature_save_dir)
                        npy_filename = os.path.join(feature_save_dir, fn.replace('.wav', f'.{s}.spec.npy'))
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
                            feature_save_dir = os.path.join(mfccs_folder, semantic_label)
                            if not os.path.isdir(feature_save_dir):
                                os.makedirs(feature_save_dir)
                            npy_filename = os.path.join(feature_save_dir, fn.replace('.wav', f'.{s}.mfcc.npy'))
                            np.save(npy_filename, mfcc)
                    ##### End: MFCCs #####
            
    print('Creating dataset done!')



def create_dataset_for_test(audio_path, melspecs_folder=None, specs_folder=None, mfccs_folder=None,
                   n_fft=1024, hop_length=512, segment_duration=3, segment_overlap=0.5, SAMPLE_RATE = 22050,DURATION = 6,sr=22050,
                   num_segments=None, n_mels=128, n_mfcc=20):

    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    if num_segments:
        num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    else:
        num_samples_per_segment = int(SAMPLE_RATE * segment_duration)
        num_segments = int((DURATION / (segment_duration * segment_overlap)) - 1)

    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) 

    for dirpath, _, filenames in os.walk(audio_path):

        if dirpath is audio_path:

            for fn in filenames:
                if os.path.splitext(fn)[-1] != '.wav':
                    continue

                print('Processing:  ',fn)

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
                        feature_save_dir = melspecs_folder
                        if not os.path.isdir(feature_save_dir):
                            os.makedirs(feature_save_dir)
                        npy_filename = os.path.join(feature_save_dir, fn.replace('.wav', f'.{s}.melspec.npy'))
                        np.save(npy_filename, melspec)
                    ##### End: MEL SPECTROGRAMS #####

                    ##### Start: SPECTROGRAMS #####
                    if specs_folder:
                        spec = librosa.core.stft(y[start_sample:finish_sample],
                                                n_fft=n_fft,
                                                hop_length=hop_length)
                        
                        spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max) 

                        # Saving the spectrogram into a .npy file
                        feature_save_dir = specs_folder
                        if not os.path.isdir(feature_save_dir):
                            os.makedirs(feature_save_dir)
                        npy_filename = os.path.join(feature_save_dir, fn.replace('.wav', f'.{s}.spec.npy'))
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
                            feature_save_dir = mfccs_folder
                            if not os.path.isdir(feature_save_dir):
                                os.makedirs(feature_save_dir)
                            npy_filename = os.path.join(feature_save_dir, fn.replace('.wav', f'.{s}.mfcc.npy'))
                            np.save(npy_filename, mfcc)
                    ##### End: MFCCs #####
            
    print('Creating dataset done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data' , help='Path for data')
    parser.add_argument('--train_test', type=str, default='train' , help='Train or test')
    args = parser.parse_args()

    assert os.path.isdir(args.data_path), 'Data path does not exist'
    audio_path = os.path.join(args.data_path, args.train_test + '_audio')
    assert os.path.isdir(audio_path), 'Audio data path does not exist'

    melspec_path = os.path.join(args.data_path, args.train_test + '_melspec')
    spec_path = os.path.join(args.data_path, args.train_test + '_spec')
    mfcc_path = os.path.join(args.data_path, args.train_test + '_mfcc')
    if not os.path.isdir(melspec_path):
        os.makedirs(melspec_path)
    if not os.path.isdir(spec_path):
        os.makedirs(spec_path)
    if not os.path.isdir(mfcc_path):
        os.makedirs(mfcc_path)

    if args.train_test == 'train':
        check_size(dataset_path=audio_path, compute_spec=True, num_segments=2)
        check_size(dataset_path=audio_path, compute_melspec=True)
        create_dataset(audio_path=audio_path, 
                    melspecs_folder=melspec_path, 
                    specs_folder=spec_path,
                    mfccs_folder=mfcc_path,
                    num_segments=2)
    else:
        #check_size_for_test(dataset_path=audio_path, compute_spec=True, num_segments=2)
        #check_size_for_test(dataset_path=audio_path, compute_melspec=True)
        create_dataset_for_test(audio_path=audio_path, 
                    melspecs_folder=melspec_path, 
                    specs_folder=spec_path,
                    mfccs_folder=mfcc_path,
                    num_segments=2)

