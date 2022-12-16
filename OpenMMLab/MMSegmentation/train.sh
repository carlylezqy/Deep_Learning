#!/bin/bash
#bash -i train.sh
cd /home/akiyo/code/openMMlab/Segmentation
MMSEG_PATH=/home/akiyo/nfs/zhang/github/mmsegmentation
python /home/akiyo/nfs/zhang/github/mmsegmentation/tools/train.py config/DRIVE_UNet.py