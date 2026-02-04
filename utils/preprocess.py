"""
Helper image/mask preprocessing utilities
"""
from PIL import Image
from typing import Tuple
import numpy as np

def resize_image(img: Image.Image, size: Tuple[int,int], is_mask: bool=False) -> Image.Image:
    """
    Resize PIL image. If is_mask=True use nearest neighbor interpolation to preserve labels.
    """
    if is_mask:
        return img.resize(size, resample=Image.NEAREST)
    else:
        return img.resize(size, resample=Image.BILINEAR)

def relabel_mask_by_color(mask: Image.Image, color_to_label: dict) -> np.ndarray:
    """
    Given a PIL RGB mask, returns a 2D numpy array where each pixel contains the label index
    from color_to_label mapping (color tuple -> label index).
    """
    arr = np.array(mask.convert("RGB"))
    h, w, _ = arr.shape
    out = np.zeros((h,w), dtype=np.uint8)
    # Build int-coded color map for speed
    flat = arr.reshape(-1,3)
    ints = (flat[:,0].astype(np.uint32) << 16) | (flat[:,1].astype(np.uint32) << 8) | flat[:,2].astype(np.uint32)
    cmap = { (c[0] << 16) | (c[1] << 8) | c[2]: idx for c, idx in color_to_label.items() }
    out_flat = np.array([ cmap.get(int(v), 0) for v in ints ], dtype=np.uint8)
    return out_flat.reshape(h,w)