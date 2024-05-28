import yaml
import torch
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from dataset.dataloader import get_dataloaders
from models.DeepLabV3Plus import model
from trainer.trainer import train_model
from utils.logger import setup_logger

config_path = 'configs/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

TRAIN_PATH = config['paths']['train']
TRAIN_GT_PATH = config['paths']['train_gt']
TEST_PATH = config['paths']['test']
log_dir = config['paths']['log_dir']
checkpoint_path = config['paths']['checkpoint']

device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
num_classes = config['training']['num_classes']
learning_rate = config['training']['learning_rate']
batch_size = config['training']['batch_size']
display_step = config['training']['display_step']
epochs = config['training']['epochs']

def get_transforms(transform_list):
    transform_objs = []
    for transform in transform_list:
        t_type = transform.pop('type')
        t_class = getattr(A, t_type)
        transform_objs.append(t_class(**transform))
    return A.Compose(transform_objs)

train_transform = get_transforms(config['transforms']['train'])
val_transform = get_transforms(config['transforms']['val'])

# Get dataloaders
train_dataloader, val_dataloader = get_dataloaders(
    TRAIN_PATH, TRAIN_GT_PATH, train_transform, val_transform, batch_size
)

# Setup model, criterion, optimizer
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()  # or your custom loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Setup logger
setup_logger(log_dir)

# Train the model
train_model(model, train_dataloader, val_dataloader, criterion, optimizer, epochs, device, checkpoint_path, log_dir)
