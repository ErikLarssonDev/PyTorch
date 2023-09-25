import torch 
from torch import nn, optim 
import os 
import config 
from torch.utils.data import DataLoader 
from tqdm import tqdm 
from sklearn.metrics import cohen_kappa_score 
from efficientnet_pytorch import EfficientNet
from dataset import DRDataset 
from torchvision.utils import save_image # To check that nothing went wrong in the pre-processing step
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    make_prediction
)

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    for batch_idx, (data, targets, _) in enumerate(loop):
    # Save examples and make sure they look ok with teh data augmentation
    # First set mean=[0, 0, 0], std=[1, 1, 1] so they look "normal"
    # save_image(data, f"hi_{batch_idx}.png")

        data = data.to(device)  
        targets = targets.to(device)

        # Forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets)

        losses.append(loss.item())

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())
    
    print(f"Loss average over epoch: {sum(losses) / len(losses)}")

def main():
    train_ds = DRDataset(
        images_folder=config.IMAGES_FOLDER,
        path_to_csv=config.PATH_TO_CSV_TRAIN,
        transform=config.train_transforms,
    )
    val_ds = DRDataset(
        images_folder=config.IMAGES_FOLDER,
        path_to_csv=config.PATH_TO_CSV_VAL,
        transform=config.val_transforms,
    )
    test_ds = DRDataset(
        images_folder=config.IMAGES_FOLDER_TEST,
        path_to_csv=config.PATH_TO_CSV_TEST,
        transform=config.val_transforms,
        train=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, num_workers=2, shuffle=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=2,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )
    loss_fn = nn.CrossEntropyLoss()
    model = EfficientNet.from_pretrained("efficientnet-b3")
    model._fc = nn.Linear(1536, 5) # Don't want 1000 classes as in imagenet, only need 5
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        print(f"EPOCH: {epoch} / {config.NUM_EPOCHS}")
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)

        # Get on validation
        preds, labels = check_accuracy(val_loader, model, config.DEVICE)
        print(f"QuadraticWeightedKappa (Validation): {cohen_kappa_score(labels, preds, weights='quadratic')}")

        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)
    
    make_prediction(model, test_loader)

if __name__ == "__main__":
    main()

    


