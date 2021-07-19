#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch.nn as nn

# 2D convolutional neural network
class ConvNet2D(nn.Module):
    def __init__(self, input_shape, nb_filters, dense_units, output_shape, 
                dropout=0.3, poolings=None):
        super(ConvNet2D, self).__init__()
        
        n_mels = input_shape[1]
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

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_shape[0], nb_filters[0], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=poolings[0], stride=2, padding=1),
            nn.BatchNorm2d(nb_filters[0]))
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(nb_filters[0], nb_filters[1], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=poolings[1], stride=2, padding=1),
            nn.BatchNorm2d(nb_filters[1]))
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(nb_filters[1], nb_filters[2], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=poolings[2], stride=2, padding=1),
            nn.BatchNorm2d(nb_filters[2]))

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(nb_filters[2], nb_filters[3], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=poolings[3], stride=2, padding=1),
            nn.BatchNorm2d(nb_filters[3]))

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(nb_filters[3], nb_filters[4], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=poolings[4], stride=2, padding=1),
            nn.BatchNorm2d(nb_filters[4]))

        self.conv_pool = nn.Sequential(
            nn.Conv2d(nb_filters[4], dense_units[0], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1))
        
        self.dense1 = nn.Sequential(
            nn.Linear(dense_units[0], dense_units[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout))
        
        self.dense2 = nn.Sequential(
            nn.Linear(dense_units[1], output_shape))
        
    def forward(self, x):
        #inpt_shape = (self.n_channels, self.input_shape[0], self.input_shape[1])
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.conv_block5(out)
        out = self.conv_pool(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.dense1(out)
        out = self.dense2(out)
        #print(out)
        return out

# 2D convolutional neural network
class ConvNet1D(nn.Module):
    def __init__(self, input_shape, nb_filters, dense_units, output_shape, 
                dropout=0.3):
        super(ConvNet1D, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_shape[0], nb_filters[0], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(nb_filters[0]))
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(nb_filters[0], nb_filters[1], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(nb_filters[1]))
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(nb_filters[1], nb_filters[2], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(nb_filters[2]))

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(nb_filters[2], nb_filters[3], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(nb_filters[3]))

        self.conv_block5 = nn.Sequential(
            nn.Conv1d(nb_filters[3], nb_filters[4], kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(nb_filters[4]))

        self.conv_pool = nn.Sequential(
            nn.Conv1d(nb_filters[4], dense_units[0], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1))
        
        self.dense1 = nn.Sequential(
            nn.Linear(dense_units[0], dense_units[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout))
        
        self.dense2 = nn.Sequential(
            nn.Linear(dense_units[1], output_shape))
        
    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.conv_block3(out)
        out = self.conv_block4(out)
        out = self.conv_block5(out)
        out = self.conv_pool(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.dense1(out)
        out = self.dense2(out)
        #print(out)
        return out

if __name__ == '__main__':
    # Create model
    model = ConvNet2D((1, 128, 130), [32,32,32,64,128], [256, 256], 3)

    # Print out all the parameters of the model
    for name, module in model.named_parameters():
        print(name, "------------", module.size())
