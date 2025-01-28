import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Thêm nhiều lớp convolution với nhiều filters
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Chuyển sang sử dụng AdaptiveAvgPool2d
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Đầu ra có kích thước 4x4 sau pooling

        # Regularization: thêm dropout
        self.dropout = nn.Dropout(0.3)  # Tăng tỷ lệ dropout lên 0.3

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # Số neurons tăng lên
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 10)  # Đầu ra là 10 lớp


    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Fourth conv block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x