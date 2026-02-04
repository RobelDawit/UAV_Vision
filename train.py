"""
Simple training script
- --prepare runs preprocessing (requires --raw-image-dir, --raw-label-dir, --class-csv)
- Otherwise trains with processed dataset in --data-dir
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from pathlib import Path
from datasets.uav_dataset import UAVDataset
from utils.data import prepare_processed_dataset

def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        out = model(imgs)['out']  # assuming torchvision segmentation model
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            out = model(imgs)['out']
            loss = criterion(out, masks)
            running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare processed dataset")
    parser.add_argument("--raw-image-dir", type=str, default=None)
    parser.add_argument("--raw-label-dir", type=str, default=None)
    parser.add_argument("--class-csv", type=str, default="class_dict_seg.csv")
    parser.add_argument("--out-dir", type=str, default="./processed")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.1)

    parser.add_argument("--data-dir", type=str, default="./processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.prepare:
        assert args.raw_image_dir and args.raw_label_dir, "Please pass raw dirs"
        prepare_processed_dataset(
            raw_image_dir=args.raw_image_dir,
            raw_label_dir=args.raw_label_dir,
            class_csv=args.class_csv,
            out_dir=args.out_dir,
            size=(args.size, args.size),
            val_split=args.val_split
        )
        print("Prepared dataset at", args.out_dir)
        return

    data_dir = Path(args.data_dir)
    train_images = data_dir / "train" / "images"
    train_masks = data_dir / "train" / "masks"
    val_images = data_dir / "val" / "images"
    val_masks = data_dir / "val" / "masks"

    train_ds = UAVDataset(str(train_images), str(train_masks))
    val_ds = UAVDataset(str(val_images), str(val_masks))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device(args.device)
    # Use torchvision FCN model (simple to swap)
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=args.num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        val_loss = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best_model.pth")
    print("Training finished.")

if __name__ == "__main__":
    main()