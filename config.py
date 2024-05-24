import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3
learning_rate = 2e-4
batch_size = 4
display_step = 50
epochs = 2