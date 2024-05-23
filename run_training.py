import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.transforms import Resize, Compose, PILToTensor, InterpolationMode, ToPILImage
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from dataset.dataloader import get_dataloaders, get_datasets

train_dataset = get_datasets()
train_set, val_set = random_split(train_dataset, [int(0.8 * len(train_dataset)), int(0.2 * len(train_dataset))])

train_dataloader, val_dataloader = get_dataloaders(train_set, val_set)

for image, label in val_dataloader:
    data = image
    target = label
    break

single_image = label[0]

plt.imshow(single_image)
plt.show()
