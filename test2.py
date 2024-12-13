import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x = self.fc2(x)                 # Linear -> logits
        return x

def test(dataloader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    test_accuracy = (correct / total) * 100 if total > 0 else 0
    print("test accuracy:", test_accuracy)

class TADataset(Dataset):
    """
    A custom dataset for the TA's directory:
    Directory structure:
      directory_name/
        test_label.txt   # each line has a single label
        test/
          image files named 0.png, 1.png, etc. corresponding to each line in test_label.txt
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        label_file = os.path.join(root_dir, "test_label.txt")
        with open(label_file, "r") as f:
            lines = f.readlines()

        # Each line contains only the label
        for i, line in enumerate(lines):
            lbl = int(line.strip())
            fname = f"{i}.png"  # Construct the filename from the index
            self.images.append(fname)
            self.labels.append(lbl)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "test", self.images[idx])
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ta", "--test_dir", type=str, default=None,
                        help="Directory containing test_label.txt and a test folder with images.")
    args = parser.parse_args()

    # Apply the same resizing and ToTensor as training (but no random rotations/affines for test)
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Resize((32, 32))
    ])

    if args.test_dir is not None:
        # Use the custom TADataset
        test_dataset = TADataset(root_dir=args.test_dir, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        # Default to MNIST if no test_dir provided
        mnist_test = MNIST(root='./MNIST_data', train=False, download=True, transform=transform)
        test_dataloader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    # Load the model
    model = torch.load("LeNet5_2.pth", map_location=device)
    model.to(device)

    test(test_dataloader, model, device)

if __name__ == "__main__":
    main()
