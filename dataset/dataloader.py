import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import Compose, Resize, InterpolationMode, PILToTensor
import torch
from torchvision import transforms

DIR = "E:/Vscode Workspace/BKAI Polyp/polyp-segmentation/data/"
TRAIN_PATH = os.path.join(DIR, 'train/train')
TRAIN_GT_PATH = os.path.join(DIR, 'train_gt/train_gt')
TEST_PATH = os.path.join(DIR, 'test/test')

class PreprocessData(Dataset):
    def __init__(self, image_path, gt_path, transform):
        super(PreprocessData, self).__init__()
        image_list = os.listdir(image_path)
        gt_list = os.listdir(gt_path)

        image_list = [os.path.join(image_path, image_name) for image_name in image_list]
        gt_list = [os.path.join(gt_path, gt_name) for gt_name in gt_list]

        self.image_list = image_list
        self.gt_list = gt_list
        self.transform = transform

    def __getitem__(self, id):
        image_path = self.image_list[id]
        gt_path = self.gt_list[id]

        data = Image.open(image_path)
        label = Image.open(gt_path)

        data = self.transform(data) / 255
        label = self.transform(label) / 255

        # Convert the mask to binary mask with threshold = 0.65
        label = torch.where(label > 0.65, 1.0, 0.0) 
        label[2, :, :] = 0.0001

        label = torch.argmax(label, 0).type(torch.int64)

        return data, label

    def __len__(self):
        return len(self.image_list)

train_transform = Compose([
    Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    PILToTensor()
])

val_transform = Compose([
    Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
    PILToTensor()
])

def get_datasets():
    train_dataset_not_aug = PreprocessData(image_path=TRAIN_PATH, gt_path=TRAIN_GT_PATH, transform=val_transform)
    train_dataset_aug = PreprocessData(image_path=TRAIN_PATH, gt_path=TRAIN_GT_PATH, transform=train_transform)
    train_dataset = ConcatDataset([train_dataset_not_aug, train_dataset_aug])
    return train_dataset

def get_dataloaders(train_set, val_set, batch_size=8):
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader