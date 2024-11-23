# Polyp Segmentation 

This repository serves as a segmentation model to classify all polyps into neoplastic or non-neoplastic classes denoted by red and green colors, respectively.

## Directory Structure
```
polyp-segmentation/
├── checkpoints/
│   ├── (model will be saved here)
├── dataset/
│   ├── dataloader.py
│   ├── preprocess.py
│   └── __init__.py
├── models/
│   ├── DeepLabV3Plus.py
│   └── unet.py
│   └── __init__.py
├── scripts/
│   ├── test.py
├── trainer/
│   ├── logger.py
│   ├── trainer.py
│   └── __init__.py
├── utils/
│   ├── mask_utils.py
│   ├── utils.py
│   ├── loss.py
│   ├── logger.py
│   └── __init__.py
├── logs/
│   └── (TensorBoard logs will be saved here)
├── run_training.py
├── config.py
├── requirements.txt
└── README.md
```



## Installation
Clone the repository and install required libraries.
  ```sh
git clone https://github.com/dvtiendat/polyp-segmentation.git
pip install -r requirements.txt
cd .\polyp-segmentation\
  ```
Ensure your data is place in the data folder

## Training
To train the model, run:
  ```sh
python run_training.py
  ```
## Testing
To visualize some of the segmentation, use:
  ```sh
python infer.py --image_path image.jpeg
  ```
## TensorBoard Visualization
To run TensorBoard, use this command:
```sh
tensorboard --logdir=logs
```
## Result
This project achieved ~0.77 accuracy in the contest of BKAI-IGH NeoPolyp
Some of the testing images:
![image](https://github.com/dvtiendat/polyp-segmentation/assets/111187020/46c6438e-470a-483c-b46d-49666609eeef)
![image](https://github.com/dvtiendat/polyp-segmentation/assets/111187020/5a121fd4-58d8-497d-8b51-4114a937f68a)
![image](https://github.com/dvtiendat/polyp-segmentation/assets/111187020/3e199c4a-2367-4e4d-8b1b-b2ef6a864d70)

## Acknowledgment
1. Dinh Sang. (2021). BKAI-IGH NeoPolyp. Kaggle. https://kaggle.com/competitions/bkai-igh-neopolyp
2. Lan, P.N., An, N.S., Hang, D.V., Long, D.V., Trung, T.Q., Thuy, N.T., Sang, D.V.: NeoUnet: Towards accurate colon polyp segmentation and neoplasm detection. In: Proceedings of the 16th International Symposium on Visual Computing (2021)
3. Nguyen Thanh Duc, Nguyen Thi Oanh, Nguyen Thi Thuy, Tran Minh Triet, Dinh Viet Sang. ColonFormer: An Efficient Transformer Based Method for Colon Polyp Segmentation. IEEE Access, vol. 10, pp. 80575-80586, 2022
4. Nguyen Hoang Thuan, Nguyen Thi Oanh, Nguyen Thi Thuy, Perry Stuart, Dinh Viet Sang (2023). RaBiT: An Efficient Transformer using Bidirectional Feature Pyramid Network with Reverse Attention for Colon Polyp Segmentation. arXiv preprint arXiv:2307.06420.
5. Nguyen Sy An, Phan Ngoc Lan, Dao Viet Hang, Dao Van Long, Tran Quang Trung, Nguyen Thi Thuy, Dinh Viet Sang. BlazeNeo: Blazing fast polyp segmentation and neoplasm detection. IEEE Access, Vol. 10, 2022.
