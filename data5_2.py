import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, RandomRotation, RandomAffine
from torch.utils.data import Dataset, DataLoader
import io

# File splits for train and test datasets
splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}

# Data augmentation and preprocessing pipeline for training
train_transform = Compose([
    Resize((32, 32)),
    # Added geometric transformations
    RandomRotation(degrees=15),                    # rotate images by +/-15deg
    RandomAffine(degrees=0, translate=(0.1, 0.1)), # small translations
    ToTensor()                                     # Convert to [0,1]
])

# For test data, we usually do less augmentation:
test_transform = Compose([
    Resize((32, 32)),
    ToTensor()  # Just convert to tensor for test
])

class MNISTDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform
        self.images, self.labels = self.preprocess_data(dataframe)

    def preprocess_data(self, df):
        images, labels = [], []
        for _, row in df.iterrows():
            # Extract image bytes and convert to PIL image
            img_data = row['image']['bytes']
            img = Image.open(io.BytesIO(img_data)).convert("L") # grayscale
            img_t = self.transform(img)
            images.append(img_t)
            labels.append(row['label'])
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_mnist_datasets(batch_size=1):
    # Load DataFrames
    df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
    df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])

    # Create datasets with different transforms
    train_dataset = MNISTDataset(df_train, transform=train_transform)
    test_dataset = MNISTDataset(df_test, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Debugging: Print sample data if run directly
if __name__ == "__main__":
    df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
    df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])

    print(df_train.head())
    print(df_test.head())
