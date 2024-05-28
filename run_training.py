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
import torchvision.utils as vutils
import segmentation_models_pytorch as smp
from models.DeepLabV3Plus import model
from utils.utils import weight_init, save_model, load_model
from config import *
from dataset.dataloader import get_dataloaders, get_datasets
from trainer.trainer import run_training

train_dataset, val_dataset = get_datasets(TRAIN_PATH, TRAIN_GT_PATH, resize=(384, 384), train_transform=train_transform, val_transform=val_transform)
train_dataloader, val_dataloader = get_dataloaders(train_dataset, val_dataset, batch_size=8)

print(model)






