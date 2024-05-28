import torch
import os
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
ROOT = 'E:/Vscode Workspace/BKAI Polyp/polyp-segmentation/'

DIR = "E:/Vscode Workspace/BKAI Polyp/polyp-segmentation/data/"
TRAIN_PATH = os.path.join(DIR, 'train/train')
TRAIN_GT_PATH = os.path.join(DIR, 'train_gt/train_gt')
TEST_PATH = os.path.join(DIR, 'test/test')

log_dir = os.path.join(ROOT, 'logs')
checkpoint = os.path.join(ROOT, 'checkpoints/model.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3
learning_rate = 2e-4
batch_size = 8
display_step = 50
epochs = 50

train_transform = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.4),
    A.RandomGamma(gamma_limit=(70, 130), p=0.2),
    A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])