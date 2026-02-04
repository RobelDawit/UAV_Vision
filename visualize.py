"""
Simple visualization tool to inspect processed images and colorized masks.
"""
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from utils.data import COARSE_PALETTE

def colorize_mask(mask_arr, palette):
    h, w = mask_arr.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    for k, col in palette.items():
        out[mask_arr==k] = col
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./processed")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--n", type=int, default=4)
    args = parser.parse_args()

    images_dir = Path(args.data_dir) / args.split / "images"
    masks_dir = Path(args.data_dir) / args.split / "masks"
    names = sorted([p.stem for p in images_dir.iterdir() if p.suffix.lower() in [".jpg",".png",".jpeg"]])[:args.n]
    for name in names:
        img = Image.open(images_dir / (name + ".jpg")).convert("RGB")
        mask = Image.open(masks_dir / (name + ".png"))
        mask_arr = np.array(mask)
        mask_color = colorize_mask(mask_arr, COARSE_PALETTE)
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(name)
        plt.subplot(1,2,2)
        plt.imshow(mask_color)
        plt.axis("off")
        plt.title("Mask (coarse)")
        plt.show()

if __name__ == "__main__":
    main()