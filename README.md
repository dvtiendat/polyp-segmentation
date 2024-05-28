# Polyp Segmentation using DeepLabV3+ model

This repository serves as a segmentation model to classify all polyps into neoplastic or non-neoplastic classes denoted by red and green colors, respectively.

## Directory Structure
```
polyp-segmentation/
│
├── configs/
│   ├── config
│
├── checkpoints/
│   ├── (model will be saved here)
│
├── dataset/
│   ├── dataloader.py
│   ├── preprocess.py
│   └── __init__.py
│
├── models/
│   ├── DeepLabV3Plus.py
│   └── unet.py
│   └── __init__.py
│
├── scripts/
│   ├── test.py
│
├── trainer/
│   ├── logger.py
│   ├── trainer.py
│   └── __init__.py
│
├── utils/
│   ├── mask_utils.py
│   ├── utils.py
│   ├── loss.py
│   ├── logger.py
│   └── __init__.py
│
├── logs/
│   └── (TensorBoard logs will be saved here)
│
├── run_training.py
├── requirements.txt
└── README.md
```


