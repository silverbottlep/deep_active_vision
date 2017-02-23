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

## Download pretrained classifier
We trained our classifier for the objects in bigbird dataset. These objects show up in the several places in the scenes. We used this classifiers to get the score of bounding boxes of the objects. This score will be the signal of training actor networks. Please refer to the paper more detail. We provide pretrained classifier that used in our paper. You can download it [here](https://drive.google.com/file/d/0B-r7apOz1BHASl94aVhDTkVqRHc/view?usp=sharing). Place this file in ./snapshots directory.

## Download and convert dataset for training actor network
Download from [project homepage](http://cs.unc.edu/~ammirato/active_vision_dataset_website), extract to some directory $(DATADIR). 
```bash
th make_datasets --data_dir $(DATADIR) --output_dir ./data
```
It will create rohit_{scene_name}.t7 files in ./data directory for each scans of the scenes. Training code will directly load the dataset from this files.

## (Optional)Navigate scenes
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

## Train actor network
```bash
th train_actor.lua --lr 0.00005 --split 1 --cnn_path ./snapshots/resnet-18.t7
```

## Test actor network
Once you have trained the actor network, you can run separate test code. you can specify the train/test splits(--split), and the number of maximum moves(--test_T)
```bash
th test_actor.lua --split 1 --test_T 5 --cnn_path ./snapshots/resnet-18.t7 2>&1 | tee split1.log
```

## Paper
A Dataset for Developing and Benchmarking Active Vision, Phil Ammirato, Patric Poirson, Eunbyung Park, Jana Kosecka, Alexander Berg, ICRA 2017

[Project Homepage](http://cs.unc.edu/~ammirato/active_vision_dataset_website)
