import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


# Custom Dataset class
class SwingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Đọc dữ liệu
bodyswing_df = pd.read_csv("BODYSWING.txt")
handswing_df = pd.read_csv("SWING.txt")

X = []
y = []
no_of_timesteps = 10

# Xử lý dữ liệu bodyswing
dataset = bodyswing_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps : i, :])
    y.append(1)

# Xử lý dữ liệu handswing
dataset = handswing_df.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(no_of_timesteps, n_sample):
    X.append(dataset[i - no_of_timesteps : i, :])
    y.append(0)

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

# Tách dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Tạo DataLoader
train_dataset = SwingDataset(X_train, y_train)
test_dataset = SwingDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Định nghĩa model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm4 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        x, _ = self.lstm3(x)
        x = self.dropout(x)
        x, _ = self.lstm4(x)
        x = self.dropout(x)
        x = self.fc(x[:, -1, :])  # Use the output of the last timestep
        return self.sigmoid(x)


# Khởi tạo model, loss, optimizer
input_dim = X.shape[2]
hidden_dim = 50
output_dim = 1

model = LSTMModel(input_dim, hidden_dim, output_dim)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {train_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")
