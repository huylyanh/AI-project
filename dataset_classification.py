import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Lambda
import os
import cv2
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt  # Import matplotlib

class FootballDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()

        self.image_paths = []
        self.labels = []
        self.categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.transform = transform

        data_path = os.path.join(root, "football")
        if train:
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "val")

        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path, str(category))
            for item in os.listdir(data_files):
                path = os.path.join(data_files, item)
                self.image_paths.append(path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB") # đọc ảnh bằng PIL
        # image = cv2.imread(image_path) #chuyen thanh numpy array
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def resize_pad(image):
    width, height = image.size
    if width > height:
        new_width = 224
        new_height = int(height * (new_width / width))
    else:
        new_height = 224
        new_width = int(width * (new_height / height))
    
    resized_image = image.resize((new_width, new_height))
    
    pad_left = (224 - new_width) // 2
    pad_right = 224 - new_width - pad_left
    pad_top = (224 - new_height) // 2
    pad_bottom = 224 - new_height - pad_top

    padded_image = Image.new("RGB", (224, 224), (0, 0, 0))
    padded_image.paste(resized_image, (pad_left, pad_top))

    return padded_image
    
if __name__ == "__main__":
    path = "data"

    transform = Compose([
        Lambda(resize_pad),
        ToTensor(),
    ])

    dataset = FootballDataset(root=path, train=True, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    sample_image, sample_label = dataset[20492]
    print("Sample image shape:", sample_image.shape)
    print("Sample label:", sample_label)    

    # Changes the order of the dimensions from (C, H, W) to (H, W, C), which is the format that matplotlib expects.
    # Converts the PyTorch tensor to a NumPy array.
    sample_image = sample_image.permute(1, 2, 0).numpy()

    plt.imshow(sample_image)
    plt.title(f"Sample Image (Label: {sample_label})")
    plt.axis("off")  # Hide axes    
    plt.show()
    

    print(len(dataloader))
    # for images, labels in dataloader:
    #     print(images.shape, labels.shape)
