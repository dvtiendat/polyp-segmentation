import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import Compose, Resize, InterpolationMode, PILToTensor
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms

class PreprocessData(Dataset):
    def __init__(self, img_path, label_path, resize=None, transform=None):
        super(PreprocessData, self).__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.resize = resize
        self.transform = transform
        self.image_list = os.listdir(self.img_path)
    
    def read_mask(self, mask_path):
        # Read the mask, resize it and convert it to HSV
        mask_image = cv2.imread(mask_path)
        mask_image = cv2.resize(mask_image, self.resize)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2HSV)

        # Constant value for color type
        lower_red1 = np.array([0, 100, 20])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160,100,20])
        upper_red2 = np.array([179,255,255])

        # Read the red mask
        lower_mask_red = cv2.inRange(mask_image, lower_red1, upper_red1)
        upper_mask_red = cv2.inRange(mask_image, lower_red2, upper_red2)
        red_mask = lower_mask_red + upper_mask_red
        red_mask[red_mask != 0] = 1

        # Read the green mask
        green_mask = cv2.inRange(mask_image, (36, 25, 25), (70, 255, 255))
        green_mask[green_mask != 0] = 2

        # Combine to get full mask
        full_mask = cv2.bitwise_or(red_mask, green_mask)
        full_mask = np.expand_dims(full_mask, axis=-1) 
        full_mask = full_mask.astype(np.uint8)
        
        return full_mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.image_list[idx])
        label_path = os.path.join(self.label_path, self.image_list[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.read_mask(label_path)
        img = cv2.resize(img, self.resize)
        if self.transform:
            img = self.transform(img)
        
        return img, label

class ProcessData(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        image = self.data[index]
        label = self.targets[index]
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image'].float()
            label = transformed['mask'].float()
            label = label.permute(2, 0, 1)
        return image, label
    
    def __len__(self):
        return len(self.data)

def get_datasets(img_path, gt_path, resize=(384, 384), train_transform=None, val_transform=None):
    dataset = PreprocessData(img_path, gt_path, resize)
    images_data = []
    labels_data = []
    for x, y in dataset:
        images_data.append(x)
        labels_data.append(y)
    
    train_size = int(0.7 * len(images_data))
    val_size = len(images_data) - train_size
    train_images = images_data[:train_size]
    train_labels = labels_data[:train_size]
    val_images = images_data[train_size:]
    val_labels = labels_data[train_size:]

    train_dataset_not_aug = ProcessData(train_images, train_labels, transform=val_transform)
    train_dataset_aug = ProcessData(train_images, train_labels, transform=train_transform)
    train_dataset = ConcatDataset([train_dataset_not_aug, train_dataset_aug])
    val_dataset = ProcessData(val_images, val_labels, transform=val_transform)
    
    return train_dataset, val_dataset

def get_dataloaders(train_set, val_set, batch_size=8):
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader

train_transform = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.4),
    A.RandomGamma (gamma_limit=(70, 130), always_apply=False, p=0.2),
    A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
