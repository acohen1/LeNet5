import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset, DataLoader
import io

# File splits for train and test datasets
splits = {'train': 'mnist/train-00000-of-00001.parquet', 'test': 'mnist/test-00000-of-00001.parquet'}

# Preprocessing pipeline
transform = Compose([
    Resize((32, 32)),
    ToTensor(),
    lambda x: -1.275 * x + 1.175
    # lambda x: x * 255.0 # Scale back to 0-255
])

# Custom Dataset class for MNIST
class MNISTDataset(Dataset):
    def __init__(self, dataframe):
        self.images, self.labels = self.preprocess_data(dataframe)

    def preprocess_data(self, df):
        images, labels = [], []
        for _, row in df.iterrows():
            # Extract image bytes and convert to PIL image
            img_data = row['image']['bytes']
            img = Image.open(io.BytesIO(img_data)).convert("L") # greyscale
            images.append(transform(img))
            labels.append(row['label'])
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Function to load data and return DataLoaders
def load_mnist_datasets(batch_size=1):
    # Load DataFrames
    df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
    df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])

    # Create datasets
    train_dataset = MNISTDataset(df_train)
    test_dataset = MNISTDataset(df_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Debugging: Print sample data when run directly
if __name__ == "__main__":
    df_train = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["train"])
    df_test = pd.read_parquet("hf://datasets/ylecun/mnist/" + splits["test"])

    print(df_train.head())
    print(df_test.head())
