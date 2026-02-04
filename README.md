# UAV Vision — Refactor of UAV_segmentation.ipynb

This repo contains a refactor of `UAV_segmentation.ipynb` into modular Python code.

Quick overview:
- utils/data.py        — class CSV reader, color→label conversion, dataset split & preprocessing pipeline
- utils/preprocess.py  — image/mask resize & relabel helpers
- datasets/uav_dataset.py — PyTorch Dataset for the processed data
- train.py             — example training script (uses torchvision segmentation model)
- visualize.py         — visualize samples and colorized masks
- requirements.txt     — Python dependencies

Default assumptions:
- Original dataset is present on disk (the original notebook used Kaggle). Set `--raw-image-dir` and `--raw-label-dir` when running preprocessing.
- Masks are RGB color-coded per `class_dict_seg.csv`. The code reads that CSV to map color → original class index.
- By default masks are remapped to 8 coarse labels. See `utils/data.py` to change mapping.

Usage example:

1. Install dependencies:
   pip install -r requirements.txt

2. Preprocess (resize and relabel):
   python train.py --prepare --raw-image-dir /path/to/original_images --raw-label-dir /path/to/label_images_semantic --out-dir ./processed --size 224 --val-split 0.1

3. Train:
   python train.py --data-dir ./processed --epochs 20 --batch-size 8

4. Visualize samples:
   python visualize.py --data-dir ./processed --n 6

If you want TensorFlow instead of PyTorch or a different coarse class mapping, tell me and I'll produce an alternative.