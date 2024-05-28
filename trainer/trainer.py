import torch
from tqdm import tqdm
from utils.logger import TensorBoardLogger

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device, checkpoint_path, log_dir):
    best_val_loss = float('inf')
    logger = TensorBoardLogger(log_dir)

    epoch_bar = tqdm(total=num_epochs, desc='Total Progress')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        num_train_batches = len(train_dataloader)

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.squeeze(dim=1).long()

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= num_train_batches

        model.eval()
        val_loss = 0.0
        num_val_batches = len(val_dataloader)
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.squeeze(dim=1).long()

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

        val_loss /= num_val_batches

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.10f}, Val Loss: {val_loss:.10f}")
        
        # Log the losses to TensorBoard
        logger.log_scalar('Train_loss', train_loss, epoch + 1)
        logger.log_scalar('Val_loss', val_loss, epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': val_loss,
            }
            torch.save(checkpoint, checkpoint_path)

        epoch_bar.update(1)

    epoch_bar.close()
    logger.close()
