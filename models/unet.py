import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.transforms import Resize, Compose, PILToTensor, InterpolationMode, ToPILImage
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.optim import lr_scheduler

class UnetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetEncoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        next_layer = self.max_pool(x)
        skip_layer = x

        return next_layer, skip_layer
    
class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDecoderBlock, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, skip_layer):
        x = self.trans_conv(x)
        x = torch.cat([x, skip_layer], axis=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class UnetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class Unet(nn.Module):
    def __init__(self, n_class=3):
        super(Unet, self).__init__()
        self.enc1 = UnetEncoderBlock(3, 64)
        self.enc2 = UnetEncoderBlock(64, 128)
        self.enc3 = UnetEncoderBlock(128, 256)
        self.enc4 = UnetEncoderBlock(256, 512)

        self.bottleneck = UnetBottleneck(512, 1024)

        self.dec1 = UnetDecoderBlock(1024, 512)
        self.dec2 = UnetDecoderBlock(512, 256)
        self.dec3 = UnetDecoderBlock(256, 128)
        self.dec4 = UnetDecoderBlock(128, 64)

        self.out = nn.Conv2d(64, n_class, kernel_size=1, padding='same')

    def forward(self, x):
        n1, sk1 = self.enc1(x)
        n2, sk2 = self.enc2(n1)
        n3, sk3 = self.enc3(n2)
        n4, sk4 = self.enc4(n3)
        n5 = self.bottleneck(n4)

        n6 = self.dec1(n5, sk4)
        n7 = self.dec2(n6, sk3)
        n8 = self.dec3(n7, sk2)
        n9 = self.dec4(n8, sk1)

        out = self.out(n9)

        return out