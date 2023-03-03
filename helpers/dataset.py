import torch
import torch.utils.data
import os
from PIL import Image


class HairStyleDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.img_labels = df.basestyle.values
        self.img_dir_path = df.img_dir_path
        self.filename = df.filename
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir_path[idx], self.filename[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.img_labels[idx]

        return image, label
