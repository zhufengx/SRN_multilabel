#!/bin/bash

###### nus_wide #####
ROOT='/home/fzhu/multi_label/datasets/nus-wide/'
IMGLIST='./datasets/nus_wide/nus_wide_test_imglist.txt'
PROTO='./models/nus_wide_resnet101_srn_test.prototxt'
MODEL='./models/nus_wide_resnet101_srn.caffemodel'
SAVENAME='./results/nus_wide_predictions.txt'

# ###### coco #####
# ROOT='/home/fzhu/multi_label/datasets/coco/'
# IMGLIST='./datasets/coco/coco_test_imglist.txt'
# PROTO='./models/coco_resnet101_srn_test.prototxt'
# MODEL='./models/coco_resnet101_srn.caffemodel'
# SAVENAME='./results/coco_predictions.txt'

# ###### wider_att #####
# ROOT='/home/fzhu/multi_label/datasets/wider_att/'
# IMGLIST='./datasets/wider_att/wider_att_test_imglist.txt'
# PROTO='./models/wider_att_resnet101_srn_test.prototxt'
# MODEL='./models/wider_att_resnet101_srn.caffemodel'
# SAVENAME='./results/wider_att_predictions.txt'


START_TIME=`date +%s`;
python ./tools/model_test.py $ROOT $IMGLIST $PROTO $MODEL $SAVENAME --gpus 0 1 2 3 4 5 6 7
END_TIME=`date +%s`;
time=$((END_TIME-START_TIME))
echo running time: $time
