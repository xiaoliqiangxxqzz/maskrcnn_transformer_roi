## Installation

### Requirements:
- PyTorch 1.0 from a nightly release. It **will not** work with 1.0 nor 1.0.1. Installation instructions can be found in https://pytorch.org/get-started/locally/
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name maskrcnn_benchmark
conda activate maskrcnn_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0 (The version of this cudatool needs to be consistent with the version of your cuda)



# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext


# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


#################
#for training
#################
1.datasets:
  1.1 for voc:
         1.mkdir datasets/voc 
	   then copy your training set and testing set into datasets/voc,and name the training set 'VOC2012' and name the testing set 'VOC2007'.  
	 2.modify this file: maskrcnn_benchmark/data/datasets/voc.py    (line 19,Modify the categories to the categories you need)
  1.2 for coco:

2.configs:
  2.1  modify this file: maskrcnn_benchmark/config/defaults.py
                         line 215   modify the _C.MODEL.ROI_BOX_HEAD.NUM_CLASSES to the num_class you need.
3.training:
  mkdir feature_1024
  python tools/train_net.py --config-file configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml    (you can choose different config-file)
#################
#for testing
#################
1.modify this file: output_z/last_checkpoint   (Change the model to the model you want to test)
2.modify this file: maskrcnn_benchmark/modeling/roi_heads/box_head/box_head.py   (line 17, Change the number of the prototypes_iter to be consistent with the model)
3.python tools/test_net.py --config-file configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml


