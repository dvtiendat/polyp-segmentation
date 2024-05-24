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

from models.unet import Unet
from utils.utils import weight_init, save_model, load_model
from config import device, num_classes, learning_rate, batch_size, display_step, epochs
from dataset.dataloader import get_dataloaders, get_datasets
from trainer.trainer import run_training

train_dataset = get_datasets()
train_set, val_set = random_split(train_dataset, [int(0.8 * len(train_dataset)), int(0.2 * len(train_dataset))])

train_dataloader, val_dataloader = get_dataloaders(train_set, val_set, batch_size=8)

model = Unet(n_class=num_classes)
model.apply(weight_init)
model = nn.DataParallel(model)
model.to(device)

weights = torch.Tensor([[0.4, 0.55, 0.05]]).to(device)
loss_fn = nn.CrossEntropyLoss(weights)
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
learning_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.6)
checkpoint_path = 'polyp-segmentation/unet_model.pth'

run_training(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_function=loss_fn,
    optimizer=optimizer,
    learning_rate_scheduler=learning_rate_scheduler,
    device=device,
    display_step=display_step,
    epochs=epochs,
    checkpoint_path=checkpoint_path
)