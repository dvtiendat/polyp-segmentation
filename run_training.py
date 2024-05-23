from dataset.dataloader import get_dataloaders, get_datasets

train_set = get_datasets()

val_dataset = train_set

train_dataloader, val_dataloader = get_dataloaders(train_set, val_dataset)

print("Train DataLoader Length:", len(train_dataloader))
print("Validation DataLoader Length:", len(val_dataloader))

for image, label in train_dataloader:
    print("Batch image shape:", image.shape)
    print("Batch label shape:", label.shape)
    break 
