import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import load_mnist_datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def fan_in_calc(layer):
    if isinstance(layer, nn.Conv2d):
        _, in_c, kH, kW = layer.weight.shape
        return in_c * kH * kW
    elif isinstance(layer, nn.Linear):
        _, in_f = layer.weight.shape
        return in_f
    else:
        return None

def init_params(model):
    # Initialize parameters with U(-2.4/Fi, 2.4/Fi)
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            Fi = fan_in_calc(layer)
            if Fi is not None:
                nn.init.uniform_(layer.weight, a=-2.4/Fi, b=2.4/Fi)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

class ModifiedLeNet5(nn.Module):
    def __init__(self):
        super(ModifiedLeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)         # C1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # use MaxPool now
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)        # C3
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # S4 now max pooling
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)      # C5
        self.fc1 = nn.Linear(120, 84)                       # F6
        self.dropout = nn.Dropout(p=0.5)                    # add dropout before the final layer
        self.fc2 = nn.Linear(84, 10)                        # Output layer for 10 classes

    def forward(self, x):
        # Use ReLU activation instead of scaled_tanh
        x = F.relu(self.conv1(x))   # C1 + activation
        x = self.pool1(x)           # S2 (max pool)
        x = F.relu(self.conv2(x))   # C3 + activation
        x = self.pool2(x)           # S4 (max pool)
        x = F.relu(self.conv3(x))   # C5 + activation
        x = x.view(-1, 120)         # Flatten
        x = F.relu(self.fc1(x))     # F6 + activation
        x = self.dropout(x)         # Dropout before output layer
        x = self.fc2(x)             # Linear output
        # We'll apply softmax in the loss function (CrossEntropyLoss does that internally)
        return x

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # With softmax-based outputs, pick argmax
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = (correct / total) * 100 if total > 0 else 0
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, criterion, epochs=10, lr=0.001):
    # Using simple GD as requested, parameter = param - lr * grad
    model.train()

    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print("NaN loss encountered. Breaking...")
                break

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param -= lr * param.grad

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = (correct / total) * 100 if total > 0 else 0
        train_loss_avg = total_loss / total if total > 0 else 0.0

        test_loss_avg, test_accuracy = evaluate(model, test_loader, criterion)

        train_loss_history.append(train_loss_avg)
        train_acc_history.append(train_accuracy)
        test_loss_history.append(test_loss_avg)
        test_acc_history.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss_avg:.4f}, Test Acc: {test_accuracy:.2f}%")

    history_df = pd.DataFrame({
        'epoch': range(1, epochs+1),
        'train_loss': train_loss_history,
        'train_accuracy': train_acc_history,
        'test_loss': test_loss_history,
        'test_accuracy': test_acc_history
    })
    history_df.to_csv('training_history.csv', index=False)
    print("Training history saved to training_history.csv")

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['epoch'], history_df['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history_df['epoch'], history_df['train_accuracy'], label='Train Acc')
    plt.plot(history_df['epoch'], history_df['test_accuracy'], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved to training_curves.png")

if __name__ == "__main__":
    train_loader, test_loader = load_mnist_datasets(batch_size=1)

    model = ModifiedLeNet5().to(device)
    init_params(model)

    # Use a standard cross-entropy loss for classification
    criterion = nn.CrossEntropyLoss()

    # Train for more epochs if desired, and consider tuning lr if needed
    train_model(model, train_loader, test_loader, criterion, epochs=20, lr=0.001)
