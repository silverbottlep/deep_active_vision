# Active Vision Implementation in Torch
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

## Navigate Scenes
You can manually navigate the scenes with following simple command
```bash
th navigate.lua --scene_name Home_01_1
```
```bash
There are 6 possible moves
1 forward
2 backward
3 left
4 right
5 rotate clockwise
6 rotate counter clockwise
```

## Train active vision 
```bash
th train_actor.lua --lr 0.00005 --split 1 --cnn_path ./snapshots/resnet-18.t7
```

## Test active vision 
Once you have trained the actor network, you can run separate test code. you can specify the train/test splits(--split), and the number of maximum moves(--test_T)
```bash
th test_actor.lua --split 1 --test_T 5 --cnn_path ./snapshots/resnet-18.t7 2>&1 | tee split1.log
```

## Paper
Please refer to the paper detail and project homepage(http://cs.unc.edu/~ammirato/active_vision_dataset_website)
```bash
A Dataset for Developing and Benchmarking Active Vision, Phil Ammirato, Patric Poirson, Eunbyung Park, Jana Kosecka, Alexander Berg, ICRA 2017
```
