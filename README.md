# GR-GAN

Pytorch implementation for reproducing GR-GAN results in the paper `GRADUAL REFINEMENT TEXT-TO-IMAGE GENERATION`

<img src="GAN_Model_V4.png" width="900px" height="550px"/>

# Usage
Get the code from github:`git clone https://github.com/BoO-18/GR-GAN.git`

Create a new conda env:`conda create -n grgan python=3.7`

Folder CLIP is code from [OPENAI](https://github.com/openai/CLIP) with some changes to the output of the image encoder and text encoder. You should run:`pip install CLIP`to install it. 

**Data**

1. Download our preprocessed metadata for [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
2. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`

**Training**
- Pre-train ITM models: `python pretrain_ITM.py --cfg cfg/ITM/coco.yml --gpu 0`
 
- Train GR-GAN models: `python main.py --cfg cfg/coco_GRAGN.yml --gpu 1`

- `*.yml` files are example configuration files for training/evaluation our models.

**Pretrained Model**
- [ITM for coco](https://pan.baidu.com/s/12MiPrA1NMteUAPzXq-XZFg?pwd=jnzz 提取码：jnzz). Download and save it to `models/`
- [GR-GAN for coco](https://pan.baidu.com/s/1pqtzFcgxcX6rl5tpCAkXjg?pwd=lbre 提取码：lbre). Download and save it to `models/`

**Demo**
- File `code/demo.ipynb` is a detailed usage example for GR-GAN.
