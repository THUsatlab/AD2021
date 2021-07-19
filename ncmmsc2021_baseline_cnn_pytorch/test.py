#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import os
import torch
import torch.nn as nn
from data import load_test_dataset
from models import ConvNet2D, ConvNet1D

def test(device, model, test_loader):
    result = []

    model.eval()  # eval mode
    with torch.no_grad():
        for features, filenames in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            result.extend(list(zip(filenames, predicted.cpu().numpy())))

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data' , help='Path for data')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cuda', type=str, default="true")

    args = parser.parse_args()

    assert os.path.isdir(args.data_path), 'Data path does not exist'

    melspec_path = os.path.join(args.data_path, 'test_melspec')
    spec_path = os.path.join(args.data_path, 'test_spec')
    mfcc_path = os.path.join(args.data_path, 'test_mfcc')
    assert os.path.isdir(melspec_path), 'melspec path does not exist'
    assert os.path.isdir(spec_path), 'spec path does not exist'
    assert os.path.isdir(mfcc_path), 'mfcc path does not exist'

    model_path = os.path.join(args.data_path, 'model')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    result_path = os.path.join(args.data_path, 'result')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    if args.cuda == "true":
        # Device configuration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    #print(device)

    # Test the model1
    print("Test the model1")
    test_loader, feature_shape = load_test_dataset(melspec_path, args.batch_size)
    model_cnn_melspec = ConvNet2D((1, feature_shape[0], feature_shape[1]), [32,32,32,64,128], [256, 256], args.num_classes)
    model_cnn_melspec.load_state_dict(torch.load(os.path.join(model_path, 'model_baseline_melspec.ckpt')))
    model_cnn_melspec = model_cnn_melspec.to(device)
    result = test(device, model_cnn_melspec, test_loader)
    result_file_name = os.path.join(result_path, 'melspec.txt')
    print("Writing to the {} file.".format(result_file_name))
    with open(result_file_name, 'w') as txt_file:
        for filename, idx in result:
            txt_file.write('{}.wav {}\n'.format(filename.split('.')[0], idx))

    # Test the model2
    print("Test the model2")
    test_loader, feature_shape = load_test_dataset(spec_path, args.batch_size)
    model_cnn_spec = ConvNet2D((1, feature_shape[0], feature_shape[1]), [32,32,32,64,128], [256, 256], args.num_classes)
    model_cnn_spec.load_state_dict(torch.load(os.path.join(model_path, 'model_baseline_spec.ckpt')))
    model_cnn_spec = model_cnn_spec.to(device)
    result = test(device, model_cnn_spec, test_loader)
    result_file_name = os.path.join(result_path, 'spec.txt')
    print("Writing to the {} file.".format(result_file_name))
    with open(result_file_name, 'w') as txt_file:
        for filename, idx in result:
            txt_file.write('{}.wav {}\n'.format(filename.split('.')[0], idx))

    # Test the model3
    print("Test the model3")
    test_loader, feature_shape = load_test_dataset(mfcc_path, args.batch_size)
    model_cnn_mfcc = ConvNet1D(feature_shape, [32,32,32,64,128], [256, 256], args.num_classes)
    model_cnn_mfcc.load_state_dict(torch.load(os.path.join(model_path, 'model_baseline_mfcc.ckpt')))
    model_cnn_mfcc = model_cnn_mfcc.to(device)
    result = test(device, model_cnn_mfcc, test_loader)
    result_file_name = os.path.join(result_path, 'mfcc.txt')
    print("Writing to the {} file.".format(result_file_name))
    with open(result_file_name, 'w') as txt_file:
        for filename, idx in result:
            txt_file.write('{}.wav {}\n'.format(filename.split('.')[0], idx))
