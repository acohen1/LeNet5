import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor, Compose
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data5_1 import load_mnist_datasets

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def scaled_tanh(x):
    # Activation: 1.7159 * tanh((2/3)*x)
    return 1.7159 * torch.tanh((2.0/3.0) * x)

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

def generate_7x12_bitmap(dataset_path, digit_class):
    folder_path = os.path.join(dataset_path, str(digit_class))
    images = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
    
    transform = Resize((7, 12))
    digit_images = []
    
    for img_path in images:
        img = Image.open(img_path).convert("L")  # grayscale
        img_tensor = transform(ToTensor()(img).unsqueeze(0))
        digit_images.append(img_tensor.numpy())
    
    mean_digit = np.mean(digit_images, axis=0)
    normalized_bitmap = 2 * mean_digit - 1
    return normalized_bitmap.squeeze()

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)         # C1
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # S2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)        # C3
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # S4
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)      # C5
        self.fc1 = nn.Linear(120, 84)                       # F6

        # Instead of a learnable linear layer, we just store RBF centers as a buffer.
        # shape: (10 classes, 84 features)
        # We'll name it rbf_centers and it will NOT be a Parameter so it won't be trained.
        self.register_buffer('rbf_centers', torch.zeros(10, 84))

    def forward(self, x):
        # Forward propagation as per LeNet-5
        x = scaled_tanh(self.conv1(x))  # C1 + activation
        x = self.pool1(x)               # S2
        x = scaled_tanh(self.conv2(x))  # C3 + activation
        x = self.pool2(x)               # S4
        x = scaled_tanh(self.conv3(x))  # C5 + activation
        x = x.view(-1, 120)             # Flatten
        x = scaled_tanh(self.fc1(x))    # F6 + activation

        # Compute RBF outputs:
        # y_i = sum_j (x_j - rbf_centers[i,j])^2 for each class i
        # x: [N,84], rbf_centers: [10,84]
        # Expand x: [N,1,84], centers: [1,10,84]
        x_expanded = x.unsqueeze(1)             # [N,1,84]
        centers = self.rbf_centers.unsqueeze(0) # [1,10,84]
        diff = x_expanded - centers             # [N,10,84]
        dist_sq = torch.sum(diff**2, dim=2)     # [N,10]
        # dist_sq is the output of the RBF layer (penalties)
        return dist_sq

def initialize_rbf_parameters(model, dataset_path):
    with torch.no_grad():
        # The original paper sets components of these parameter vectors to +1 or -1.
        # The provided generate_7x12_bitmap function returns values in [-1,1].
        # Flatten the bitmap and store in rbf_centers.
        for digit_class in range(10):
            bitmap = generate_7x12_bitmap(dataset_path, digit_class)
            flattened_bitmap = torch.tensor(bitmap.flatten(), dtype=torch.float32, device=device)
            # Assign this vector to the corresponding row in rbf_centers
            model.rbf_centers[digit_class] = flattened_bitmap

class CustomLossEq9(nn.Module):
    def __init__(self, j=0.1):
        super(CustomLossEq9, self).__init__()
        self.j = torch.tensor(j, dtype=torch.float32)

    def forward(self, outputs, targets):
        # outputs are the RBF penalties [N, C]
        # Following eq(9), we have:
        # E(W) = (1/P)*sum_p [ y_correct + log(e^{-j} + sum_i e^{-y_i}) ]
        # We exclude the correct class from the sum inside the log term.
        batch_size = outputs.size(0)
        device = outputs.device
        j_tensor = self.j.to(device)

        targets = targets.long()
        y_correct = outputs[torch.arange(batch_size, device=device), targets]

        neg_j = torch.full((batch_size, 1), -j_tensor.item(), device=device)

        neg_y = -outputs
        mask = torch.ones_like(neg_y, dtype=torch.bool)
        mask[torch.arange(batch_size), targets] = False  # exclude correct class
        neg_y_incorrect = neg_y[mask].view(batch_size, -1)

        all_values = torch.cat([neg_j, neg_y_incorrect], dim=1)
        log_term = torch.logsumexp(all_values, dim=1)

        loss = (y_correct + log_term).mean()
        return loss

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
            # Prediction is the class with smallest penalty:
            # because the RBF output is a penalty, the correct class should be minimal
            predicted = torch.argmin(outputs, dim=1)
            correct += (predicted == labels.long()).sum().item()
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
            predicted = torch.argmin(outputs, dim=1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_mat[t.item(), p.item()] += 1
    return confusion_mat

def train_model(model, train_loader, test_loader, criterion, epochs=10, lr=0.001):
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
        
        for i, (inputs, labels) in enumerate(train_loader):
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
            predicted = torch.argmin(outputs, dim=1)
            correct += (predicted == labels.long()).sum().item()
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
    history_df.to_csv('training_history5_1.csv', index=False)
    print("Training history saved to training_history5_1.csv")

    torch.save(model, "LeNet5_1.pth")
    print("Model state dict saved to LeNet5_1.pth")

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
    plt.savefig('training_curves5_1.png')
    print("Training curves saved to training_curves5_1.png")

    confusion_mat = compute_confusion_matrix(model, test_loader)
    confusion_df = pd.DataFrame(
        confusion_mat,
        index=[f"True_{i}" for i in range(10)],
        columns=[f"Pred_{i}" for i in range(10)]
    )
    confusion_df.to_csv("confusion_matrix5_1.csv", index=True)
    print("Confusion matrix saved to confusion_matrix5_1.csv")

if __name__ == "__main__":
    train_loader, test_loader = load_mnist_datasets(batch_size=1)

    model = LeNet5().to(device)
    init_params(model)
    initialize_rbf_parameters(model, "digits_jpeg")
    
    criterion = CustomLossEq9(j=0.1)

    train_model(model, train_loader, test_loader, criterion, epochs=10, lr=0.001)
