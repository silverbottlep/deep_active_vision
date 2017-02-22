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

## Train active vision 
```bash
th train_actor.lua
```

## Test active vision 
```bash
th test_actor.lua
```
