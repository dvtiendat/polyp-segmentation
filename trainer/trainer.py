import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import save_model

def train(train_dataloader, valid_dataloader, model, optimizer, loss_function, learning_rate_scheduler, epoch, device, display_step):
    print(f"Start epoch #{epoch+1}, learning rate for this epoch: {learning_rate_scheduler.get_last_lr()}")
    train_loss_epoch = 0
    test_loss_epoch = 0
    model.train()

    for i, (data, targets) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch")):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_function(outputs, targets.long())
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
        if (i + 1) % display_step == 0:
            print(f'Train Epoch: {epoch + 1} [{(i + 1) * len(data)}/{len(train_dataloader.dataset)} ({100 * (i + 1) / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.4f}')

    train_loss_epoch /= len(train_dataloader)

    model.eval()
    with torch.no_grad():
        for data, target in valid_dataloader:
            data, target = data.to(device), target.to(device)
            test_output = model(data)
            test_loss = loss_function(test_output, target)
            test_loss_epoch += test_loss.item()

    test_loss_epoch /= len(valid_dataloader)
    return train_loss_epoch, test_loss_epoch

def plot_loss(train_loss_array, test_loss_array):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_array, label='Train Loss')
    plt.plot(test_loss_array, label='Test Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def run_training(model, train_dataloader, val_dataloader, loss_function, optimizer, learning_rate_scheduler, device, display_step, epochs, checkpoint_path):
    train_loss_array = []
    test_loss_array = []
    last_loss = float('inf')

    for epoch in range(epochs):
        train_loss_epoch, test_loss_epoch = train(
            train_dataloader, val_dataloader, model, optimizer, loss_function, learning_rate_scheduler, epoch, device, display_step
        )

        if test_loss_epoch < last_loss:
            save_model(model, optimizer, checkpoint_path)
            last_loss = test_loss_epoch

        learning_rate_scheduler.step()
        train_loss_array.append(train_loss_epoch)
        test_loss_array.append(test_loss_epoch)

    print("Train Loss Array:", train_loss_array)
    print("Test Loss Array:", test_loss_array)
    plot_loss(train_loss_array, test_loss_array)