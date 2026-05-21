# src/model.py

import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    """
    A Convolutional Neural Network for emotion classification.
    """
    def __init__(self):
        super(EmotionCNN, self).__init__()
        # Input: 1x48x48 grayscale image
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully Connected Classifier
        # Input size = 128 channels * 6x6 image (after 3 pooling layers from 48x48)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc_bn = nn.BatchNorm1d(128)
        self.fc_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 7) # 7 emotion classes

    def forward(self, x):
        # Pass through convolutional blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        # Flatten the feature maps
        x = x.view(-1, 128 * 6 * 6)
        
        # Pass through the classifier
        x = F.relu(self.fc_bn(self.fc1(x)))
        x = self.fc_dropout(x)
        x = self.fc2(x)
        
        return x