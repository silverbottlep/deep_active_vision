# Active vision implementation in torch
=======================================

## Download facebook resnet implementation and pretrained models
Download from https://github.com/facebook/fb.resnet.torch and install it into current directory
```bash
git clone https://github.com/facebook/fb.resnet.torch.git
```

Download pretrained model resnet-18
```bash
wget -P snapshots/ https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7
```

## Download Pretrained Classifier
We trained our classifier on bigbird dataset. We provide pretrained classifier that used in our paper. You can download it via 

## Download and Convert Dataset for training actor network
Download from http://cs.unc.edu/~ammirato/active_vision_dataset_website, extract to some directory$(DATADIR). 
```bash
th make_datasets --data_dir $(DATADIR) --output_dir ./data
```
It will create rohit_$(scene_name).t7 file for each scans of the scenes. Training code will directly load the dataset from this file.

## Train active vision 
```bash
th train_actor.lua --lr 0.00005 --split 1 --cnn_path ./snapshots/resnet-18.t7
```

## Test active vision 
```bash
th test_actor.lua
```
