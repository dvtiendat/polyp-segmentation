# test.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.dataloader import get_datasets
from models.DeepLabV3Plus import model
from utils.mask_util import mask_to_rgb, color_dict
from config import *

def visualize_batch(images, ground_truths, predictions, batch_size=8):
    fig, ax = plt.subplots(batch_size, 3, figsize=(20, 5 * batch_size))
    for i in range(batch_size):
        ax[i, 0].imshow(images[i])
        ax[i, 0].set_title("Original Image")
        ax[i, 0].axis("off")
        ax[i, 1].imshow(ground_truths[i])
        ax[i, 1].set_title("Ground Truth Mask")
        ax[i, 1].axis("off")
        ax[i, 2].imshow(predictions[i])
        ax[i, 2].set_title("Predicted Mask")
        ax[i, 2].axis("off")
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(checkpoint)['model'])
model.to(device)
model.eval()

train_dataset, val_dataset = get_datasets(TRAIN_PATH, TRAIN_GT_PATH, resize=(384, 384), train_transform=train_transform, val_transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)  

count = 0
for images, labels in val_dataloader:
    images = images.to(device)
    labels = labels.squeeze(dim=1).cpu().numpy()

    with torch.no_grad():
        output_masks = model(images).cpu().numpy().transpose(0, 2, 3, 1)

    batch_size = images.size(0)
    ori_images = []
    gt_masks = []
    pred_masks = []

    for i in range(batch_size):
        ori_h, ori_w = labels[i].shape[0], labels[i].shape[1]
        mask = cv2.resize(output_masks[i], (ori_w, ori_h))
        mask = np.argmax(mask, axis=2)
        mask_rgb = mask_to_rgb(mask, color_dict)

        ori_img = images[i].cpu().numpy().transpose(1, 2, 0)
        ori_img = (ori_img * 255).astype(np.uint8)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)

        gt_mask = labels[i]
        gt_mask_rgb = mask_to_rgb(gt_mask, color_dict)

        ori_images.append(ori_img)
        gt_masks.append(gt_mask_rgb)
        pred_masks.append(mask_rgb)

    visualize_batch(ori_images, gt_masks, pred_masks, batch_size)

    count += 1
    if count == 1:
        break
