"""
Dataset utility functions:
- read_class_dict: reads the CSV with colors and names
- rgb_to_index: convert RGB mask to original class indices
- coarse mapping: default mapping from 24 classes -> 8 classes (adjustable)
- prepare_processed_dataset: resize masks/images, relabel to coarse classes, save to folder
"""
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from .preprocess import resize_image

def read_class_dict(csv_path: str) -> List[Dict]:
    """
    Reads the class CSV containing columns: name, r, g, b
    Returns list of dicts in file order; indices correspond to original class ids (0..23).
    """
    df = pd.read_csv(csv_path)
    classes = []
    for _, row in df.iterrows():
        classes.append({
            "name": row["name"],
            "color": (int(row["r"]), int(row["g"]), int(row["b"]))
        })
    return classes

# Default coarse mapping (24 -> 8). Adjust as needed.
# The index keys are the original class indices (0..23) as in class_dict_seg.csv
# Values are new class ids in range 0..7
DEFAULT_COARSE_MAP = {
    # 0 unlabeled -> 0 background
    0: 0,
    # built / paved / roof / wall / gravel / dirt / rocks -> 1 built
    1: 1, 2: 1, 4: 1, 9: 1, 10: 1, 6: 1,
    # grass / vegetation / tree / bald-tree -> 2 vegetation
    3: 2, 8: 2, 19: 2, 20: 2,
    # water / pool -> 3 water
    5: 3, 7: 3,
    # vehicle classes -> 4 vehicle
    17: 4, 18: 4,
    # person / dog -> 5 person
    15: 5, 16: 5,
    # fences / obstacle -> 6 barrier
    13: 6, 14: 6, 22: 6,
    # other (window, door, ar-marker, conflicting) -> 7 other
    11: 7, 12: 7, 21: 7, 23: 7
}

# Fill any unmapped original indices by mapping them to `7` (other)
def ensure_full_map(coarse_map):
    full = dict(coarse_map)
    for i in range(24):
        if i not in full:
            full[i] = 7
    return full

DEFAULT_COARSE_MAP = ensure_full_map(DEFAULT_COARSE_MAP)

# Representative colors for the coarse classes (for visualization)
COARSE_PALETTE = {
    0: (0, 0, 0),        # background
    1: (128, 64, 128),   # built
    2: (0, 128, 0),      # vegetation
    3: (28, 42, 168),    # water
    4: (9, 143, 150),    # vehicle
    5: (255, 22, 96),    # person
    6: (190, 153, 153),  # barrier
    7: (255, 255, 0)     # other
}

def rgb_to_index(mask_rgb: np.ndarray, classes: List[Dict]) -> np.ndarray:
    """
    Convert an (H,W,3) uint8 RGB mask into an (H,W) int array of original class indices.
    classes is the list returned by read_class_dict (index matches original class id).
    """
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)
    # Build color -> index map
    color_map = {cls["color"]: idx for idx, cls in enumerate(classes)}
    # Convert mask to tuples and map
    flat = mask_rgb.reshape(-1, 3)
    ints = (flat[:,0].astype(np.uint32) << 16) | (flat[:,1].astype(np.uint32) << 8) | flat[:,2].astype(np.uint32)
    color_to_idx_int = { (c[0] << 16) | (c[1] << 8) | c[2]: idx for c, idx in color_map.items() }
    out_flat = np.zeros(ints.shape[0], dtype=np.uint8)
    # Default unknown colors -> 0 (background)
    for i, val in enumerate(ints):
        out_flat[i] = color_to_idx_int.get(int(val), 0)
    return out_flat.reshape(h, w)

def prepare_processed_dataset(
    raw_image_dir: str,
    raw_label_dir: str,
    class_csv: str,
    out_dir: str,
    size: Tuple[int,int] = (224,224),
    val_split: float = 0.1,
    coarse_map: Dict[int,int] = DEFAULT_COARSE_MAP
):
    """
    Prepares processed dataset:
    - reads original images and RGB masks
    - resizes images and masks to `size`
    - relabels mask colors -> original indices -> coarse mapping
    - saves into out_dir/train/images, out_dir/train/masks, out_dir/val/...
    """
    classes = read_class_dict(class_csv)
    raw_image_dir = Path(raw_image_dir)
    raw_label_dir = Path(raw_label_dir)
    out_dir = Path(out_dir)
    train_img_dir = out_dir / "train" / "images"
    train_mask_dir = out_dir / "train" / "masks"
    val_img_dir = out_dir / "val" / "images"
    val_mask_dir = out_dir / "val" / "masks"
    for p in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        p.mkdir(parents=True, exist_ok=True)
    image_files = sorted([p for p in raw_image_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    label_files = sorted([p for p in raw_label_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    # match images and labels by stem
    image_map = {p.stem: p for p in image_files}
    label_map = {p.stem: p for p in label_files}
    common = sorted(set(image_map.keys()) & set(label_map.keys()))
    n_val = max(1, int(len(common) * val_split))
    val_names = set(common[:n_val])
    for name in tqdm(common, desc="Processing"):
        img_path = image_map[name]
        mask_path = label_map[name]
        img_out_dir = val_img_dir if name in val_names else train_img_dir
        mask_out_dir = val_mask_dir if name in val_names else train_mask_dir
        # Load image and mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        # Resize
        img_resized = resize_image(img, size)
        mask_resized = resize_image(mask, size, is_mask=True)
        # Convert mask to original indices
        mask_arr = np.array(mask_resized)
        orig_idx = rgb_to_index(mask_arr, classes)
        # Map to coarse
        coarse_idx = np.vectorize(lambda x: coarse_map.get(int(x), 7))(orig_idx).astype(np.uint8)
        # Save
        img_resized.save(img_out_dir / (name + ".jpg"), format="JPEG", quality=95)
        # Save masks as single-channel PNG
        Image.fromarray(coarse_idx).save(mask_out_dir / (name + ".png"), format="PNG")