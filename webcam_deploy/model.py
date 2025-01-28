import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 32 filters, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 64 filters, 3x3 kernel
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPooling layer
        
        # Tính kích thước đầu ra
        # Input ban đầu: (3, 224, 224)
        # Sau Conv1 + Pool: (32, 112, 112)
        # Sau Conv2 + Pool: (64, 56, 56)
        self.feature_size = 64 * 56 * 56

        self.fc1 = nn.Linear(self.feature_size, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes)   # Output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Convolution 1 + ReLU
        x = self.pool(x)           # MaxPooling
        x = F.relu(self.conv2(x))  # Convolution 2 + ReLU
        x = self.pool(x)           # MaxPooling
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))    # Fully connected layer 1 + ReLU
        x = self.fc2(x)            # Output layer
        return x