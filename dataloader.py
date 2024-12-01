import torch
import os
from PIL import Image
import pandas as pd

class GTSRBTrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        for class_dir in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                annotation_file = os.path.join(class_path, f"GT-{class_dir}.csv")
                annotations = pd.read_csv(annotation_file, sep=";")
                for _, row in annotations.iterrows():
                    img_path = os.path.join(class_path, row["Filename"])
                    self.data.append((img_path, int(row["ClassId"])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class GTSRBTestDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file, sep=";")
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.annotations.iloc[idx, -1])
        if self.transform:
            image = self.transform(image)
        return image, label
    
