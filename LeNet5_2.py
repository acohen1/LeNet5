import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from data5_2 import load_mnist_datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# A modernized LeNet-like model:
# - Uses ReLU instead of scaled_tanh
# - Uses MaxPool instead of AvgPool
# - Includes a Dropout layer before the final fully-connected layer
# - Outputs class logits directly, no RBF or custom loss needed
class ModernLeNet(nn.Module):
    def __init__(self):
        super(ModernLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(84, 10)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # Convolution + ReLU
        x = self.pool(x)                # MaxPool
        x = F.relu(self.conv2(x))       # Convolution + ReLU
        x = self.pool(x)                # MaxPool
        x = F.relu(self.conv3(x))       # Convolution + ReLU
        x = x.view(-1, 120)             # Flatten
        x = F.relu(self.fc1(x))         # Fully connected + ReLU
        x = self.dropout(x)             # Dropout
        x = self.fc2(x)                 # Final linear layer -> logits
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
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = (correct / total) * 100 if total > 0 else 0
    return avg_loss, accuracy

def compute_confusion_matrix(model, loader):
    model.eval()
    num_classes = 10
    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_mat[t.item(), p.item()] += 1
    return confusion_mat

def train_model(model, train_loader, test_loader, criterion, epochs=10, lr=0.001):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

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

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = (correct / total) * 100
        train_loss_avg = total_loss / total

        test_loss_avg, test_accuracy = evaluate(model, test_loader, criterion)

        train_loss_history.append(train_loss_avg)
        train_acc_history.append(train_accuracy)
        test_loss_history.append(test_loss_avg)
        test_acc_history.append(test_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss_avg:.4f}, Test Acc: {test_accuracy:.2f}%")

    history_df = pd.DataFrame({
        'epoch': range(1, epochs+1),
        'train_loss': train_loss_history,
        'train_accuracy': train_acc_history,
        'test_loss': test_loss_history,
        'test_accuracy': test_acc_history
    })
    history_df.to_csv('training_history5_2.csv', index=False)
    print("Training history saved to training_history5_2.csv")

    torch.save(model, "LeNet5_2.pth")
    print("Model state dict saved to LeNet5_2.pth")

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
    plt.savefig('training_curves5_2.png')
    print("Training curves saved to training_curves.png")

    confusion_mat = compute_confusion_matrix(model, test_loader)
    confusion_df = pd.DataFrame(
        confusion_mat,
        index=[f"True_{i}" for i in range(10)],
        columns=[f"Pred_{i}" for i in range(10)]
    )
    confusion_df.to_csv("confusion_matrix5_2.csv", index=True)
    print("Confusion matrix saved to confusion_matrix5_2.csv")

if __name__ == "__main__":
    train_loader, test_loader = load_mnist_datasets(batch_size=64)
    model = ModernLeNet().to(device)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, test_loader, criterion, epochs=20, lr=0.01)