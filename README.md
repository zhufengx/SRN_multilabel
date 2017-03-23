# Spatial Regularization Network
This is the demo code for 

Feng Zhu, Hongsheng Li, Wanli Ouyang, Nenghai Yu, Xiaogang Wang, "Learning Spatial Regularization with Image-level Supervisions for Multi-label Image Classification", CVPR 2017. [pdf](https://arxiv.org/abs/1702.05891). 

## Directories and Files

- `caffe/`: an early version of Yuanjun Xiong's [caffe](https://github.com/yjxiong/caffe), with OpenMPI-based Multi-GPU support.
- `tools/`: demo code for model testing and evaluation.
- `run_test.sh`: script for model testing.
- `evaluation.m`: matlab code for classification results evaluation.

## Prepare data

- Download [train/test split files](https://drive.google.com/file/d/0B7lJth6WXHffQzdDUlc2NjNOUWM/view) for NUS-WIDE, MS-COCO and WIDR-Attribute, and extract it to `datasets/`.
- Download dataset images
	* [NUS-WIDE](http://lms.comp.nus.edu.sg/research/NUS-WIDE.htm): this dataset contains many untagged images, and some download links are invalid now. By removing invalid and untagged images, we finally get 119,986 images for training and 80,283 images for testing.
	* [MS-COCO_2014](http://mscoco.org/dataset/#download): 82,783 images in "train2014" for training, and 40,504 images in "val2014" for testing.
	* WIDER-Attribute: original images are provided [here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERAttribute.html), cropped images for each human bounding box can be downloaded [here](https://drive.google.com/open?id=0B7lJth6WXHffZEZFVEk5M0w3bDA). 28,340 cropped images in "train" and "val" for training, 29,177 cropped images in "test" for testing.
- Download trained models, and extract it to `models/`
	* [models](https://drive.google.com/open?id=0B7lJth6WXHffUTByUFZqNFhTaWM), containing: 
		- trained models for NUS-WIDE, MS-COCO and WIDR-Attribute.
		- a ResNet-101 model pretrained on ImageNet.
- (Optional) Download [reference classification results](https://drive.google.com/open?id=0B7lJth6WXHffc0NGSmJidkNjS2M), and extract it to `results/`.

## Build ##

See Yuanjun Xiong's [github](https://github.com/yjxiong/caffe) for building this version of caffe.

## Run Test 

- Edit `run_test.sh`
	* uncomment to specify settings of one dataset.
	* modify variable "ROOT": the root directory holding images of each dataset.
	* modify parameters of "--gpus" to specify available gpus for testing.
- Edit `tools/model_test.py`
	* add "path to your caffe" to the search path of python at line 4.

## Evaluation

- Specify one dataset in line 4 of `evaluation.m`.
- Default settings will evaluate [reference classification results](https://drive.google.com/open?id=0B7lJth6WXHffc0NGSmJidkNjS2M). 
