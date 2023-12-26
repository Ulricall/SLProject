import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.layer3 = nn.Linear(hidden_sizes[1], num_classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class DeepLearningModel:
    def __init__(self, hidden=[256, 256], lr=0.01):
        self.model = NeuralNet(input_size=733, hidden_sizes=hidden, num_classes=20)
        self.lr = lr
    
    def fit(self, X, y):
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        # print(X_tensor.shape)

        for epoch in range(10):
            for i, (inputs, labels) in enumerate(train_loader):
                # Forward pass
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()
    
    def compute_loss(self, X, y):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        # print(X.shape)
        with torch.no_grad():
            outputs = self.model(X)
            loss = F.cross_entropy(outputs, y, reduction='sum').item()
        self.model.train()
        return loss
    
    def compute_aic(self, X, y):
        nll = self.compute_loss(X, y)
        k = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print(k)
        aic = 2 * k + 2 * nll
        return aic
    
    def compute_bic(self, X, y):
        n_samples = X.shape[0]
        nll = self.compute_loss(X, y)
        k = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        bic = k * np.log(n_samples) + 2 * nll
        return bic