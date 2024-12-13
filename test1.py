import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scaled_tanh(x):
    # Activation: 1.7159 * tanh((2/3)*x)
    return 1.7159 * torch.tanh((2.0/3.0) * x)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)         # C1
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # S2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)        # C3
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # S4
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)      # C5
        self.fc1 = nn.Linear(120, 84)                       # F6

        # RBF centers buffer
        self.register_buffer('rbf_centers', torch.zeros(10, 84))

    def forward(self, x):
        x = scaled_tanh(self.conv1(x))  # C1 + activation
        x = self.pool1(x)               # S2
        x = scaled_tanh(self.conv2(x))  # C3 + activation
        x = self.pool2(x)               # S4
        x = scaled_tanh(self.conv3(x))  # C5 + activation
        x = x.view(-1, 120)             # Flatten
        x = scaled_tanh(self.fc1(x))    # F6 + activation

        # Compute RBF outputs (penalties)
        x_expanded = x.unsqueeze(1)             # [N,1,84]
        centers = self.rbf_centers.unsqueeze(0) # [1,10,84]
        diff = x_expanded - centers             # [N,10,84]
        dist_sq = torch.sum(diff**2, dim=2)     # [N,10]
        return dist_sq

def test(dataloader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Using argmin since outputs are penalties
            predictions = torch.argmin(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    test_accuracy = (correct / total) * 100 if total > 0 else 0
    print("test accuracy:", test_accuracy)

def main():
    # Use Pad and ToTensor transforms
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        lambda x: -1.275 * x + 1.175
    ])

    # Load MNIST test set
    mnist_test = MNIST(root='./MNIST_data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    # Load the entire model
    model = torch.load("LeNet5_1.pth", map_location=device)
    model.to(device)

    test(test_dataloader, model, device)

if __name__ == "__main__":
    main()