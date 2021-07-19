#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import os
import torch
import torch.nn as nn
from data import load_train_dataset
from models import ConvNet2D, ConvNet1D

def train(device, model, train_loader, num_epochs, lr):
    model.train()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

def validate(device, model, val_loader):
    model.eval()  # eval mode
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the val data: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data' , help='Path for data')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--cuda', type=str, default="true")

    args = parser.parse_args()

    assert os.path.isdir(args.data_path), 'Data path does not exist'

    melspec_path = os.path.join(args.data_path, 'train_melspec')
    spec_path = os.path.join(args.data_path, 'train_spec')
    mfcc_path = os.path.join(args.data_path, 'train_mfcc')
    assert os.path.isdir(melspec_path), 'melspec path does not exist'
    assert os.path.isdir(spec_path), 'spec path does not exist'
    assert os.path.isdir(mfcc_path), 'mfcc path does not exist'

    model_path = os.path.join(args.data_path, 'model')
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    if args.cuda == "true":
        # Device configuration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    #print(device)

    # Train the model1
    train_loader, val_loader, feature_shape = load_train_dataset(melspec_path, args.batch_size)
    model_cnn_melspec = ConvNet2D((1, feature_shape[0], feature_shape[1]), [32,32,32,64,128], [256, 256], args.num_classes)
    model_cnn_melspec = model_cnn_melspec.to(device)
    train(device, model_cnn_melspec, train_loader, args.num_epochs, args.learning_rate)
    validate(device, model_cnn_melspec, val_loader)
    torch.save(model_cnn_melspec.state_dict(), os.path.join(model_path, 'model_baseline_melspec.ckpt'))

    # Train the model2
    train_loader, val_loader, feature_shape = load_train_dataset(spec_path, args.batch_size)
    model_cnn_spec = ConvNet2D((1, feature_shape[0], feature_shape[1]), [32,32,32,64,128], [256, 256], args.num_classes)
    model_cnn_spec = model_cnn_spec.to(device)
    train(device, model_cnn_spec, train_loader, args.num_epochs, args.learning_rate)
    validate(device, model_cnn_spec, val_loader)
    torch.save(model_cnn_spec.state_dict(), os.path.join(model_path, 'model_baseline_spec.ckpt'))

    # Train the model3
    train_loader, val_loader, feature_shape = load_train_dataset(mfcc_path, args.batch_size)
    model_cnn_mfcc = ConvNet1D(feature_shape, [32,32,32,64,128], [256, 256], args.num_classes)
    model_cnn_mfcc = model_cnn_mfcc.to(device)
    train(device, model_cnn_mfcc, train_loader, args.num_epochs, args.learning_rate)
    validate(device, model_cnn_mfcc, val_loader)
    torch.save(model_cnn_mfcc.state_dict(), os.path.join(model_path, 'model_baseline_mfcc.ckpt'))
