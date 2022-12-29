# ResNet50_VOC2012.py
_base_ = ['voc2012.py', ...]
inc_name = "GHELIA_INC."
train_config = dict(model="resnet50", num_classes=20)