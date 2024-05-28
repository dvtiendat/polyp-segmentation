import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_softmax = F.softmax(inputs, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        inputs_flat = inputs_softmax.contiguous().view(-1)
        targets_flat = targets_one_hot.contiguous().view(-1)

        intersection = (inputs_flat * targets_flat).sum()

        dice = (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)

        dice_loss = 1 - dice

        return dice_loss