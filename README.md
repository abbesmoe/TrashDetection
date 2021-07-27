<p align="center">
<img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/logonav.png" width="25%"/>
</p>

TACO is a growing image dataset of waste in the wild. It contains images of litter taken under
diverse environments: woods, roads and beaches. These images are manually labeled and segmented
according to a hierarchical taxonomy to train and evaluate object detection algorithms. Currently,
images are hosted on Flickr and we have a server that is collecting more images and
annotations @ [tacodataset.org](http://tacodataset.org)


<div align="center">
  <div class="column">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/1.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/2.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/3.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/4.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/5.png" width="17%" hspace="3">
  </div>
</div>
</br>

For convenience, annotations are provided in COCO format. Check the metadata here:
http://cocodataset.org/#format-data

TACO is still relatively small, but it is growing. Stay tuned!

# Publications

For more details check our paper: https://arxiv.org/abs/2003.06975

If you use this dataset and API in a publication, please cite us using: &nbsp;
```
@article{taco2020,
    title={TACO: Trash Annotations in Context for Litter Detection},
    author={Pedro F Proença and Pedro Simões},
    journal={arXiv preprint arXiv:2003.06975},
    year={2020}
}
```

# News
**December 20, 2019** - Added more 785 images and 2642 litter segmentations. <br/>
**November 20, 2019** - TACO is officially open for new annotations: http://tacodataset.org/annotate

# Getting started

### Requirements
* python 3.7
* tensorflow 2.5
* keras 2.5

### How to setup
1. Open Google CoLab and create a new notebook
2. Clone the repository 
``` !git clone https://github.com/abbesmoe/TrashDetection ```
3. Install the requred packages
``` 
!pip install keras==2.5.0rc0
!pip install tensorflow==2.5
!pip install 'h5py==2.10.0'
```
4. Change the directory 
``` %cd TrashDetection/ ```
5. Import the necessary libraries
``` 
%matplotlib inline
import csv
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io
from detector import dataset
from detector import model as modellib
from detector import visualize
from detector.config import Config
from os import path
from PIL import Image
```
6. Check the number of images used and print the classes
```
class_map = {}
with open("detector/taco_config/map_10.csv") as csvfile:
    reader = csv.reader(csvfile)
    class_map = {row[0]:row[1] for row in reader}

TACO_DIR = "data/"
round = None 
subset = "train"
dataset = dataset.Taco()
taco = dataset.load_taco(TACO_DIR, round, subset,
                         class_map=class_map, return_taco=True)

dataset.prepare()

print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))
```
7. Create confgi class and override hyperparamters
```
class TacoTestConfig(Config):
  NAME = "taco"
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1
  NUM_CLASSES = dataset.num_classes

config = TacoTestConfig()
config.display()
```
8. Create the model and load pretrained weights
 ```
 model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_taco.hy',
                          config=config)

 model.load_weights('mask_rcnn_taco_0100.h5', by_name=True,
                   weights_out_path=None)
```
Note: if you recieve the following the error: "OSError: Unable to open file (file signature not found)", delete the h5 file and redownload it. You can download the h5 file from the following link, https://github.com/pedropro/TACO/releases/tag/1.0. Scroll to the bottom of the page and download "taco_10_3.zip".

9. Create list of class names
```
class_names = ["BG","Bottle","Bottle cap","Can","Cigarette","Cup",
               "Lid","Other","Plastic bag + wrapper","Pop tab","Straw"]
```
10. Upload your own images into the static/uploads folder. Open image, and convert images to .jpg.
```
imagePath = "static/uploads/"
imageName = "Your image.jpg"
img = Image.open(os.path.join(imagePath, imageName))

fileName = os.path.splitext(imageName)[0]
fileExtension = os.path.splitext(imageName)[1]

if fileExtension != '.jpg':
  img.convert('RGB').save("{}{}.jpg".format(imagePath, fileName))
  imageName = "{}.jpg".format(fileName)

image = skimage.io.imread(os.path.join(imagePath, imageName))
```
11. Run the detection
```
r = model.detect([image], verbose=0)[0]
```
12. define the minimum accuracy displayed after detection.
```
def min_accuracy(r,a):
  result = {'rois': [], 'masks': [], 'class_ids': [], 'scores': []}
  indecies = []
  for i,ele in enumerate(r['scores']):
    if ele >= a:
      result['rois'].append(r['rois'][i])
      result['class_ids'].append(r['class_ids'][i])
      result['scores'].append(r['scores'][i])
      indecies.append(i)

  result['masks'] = r['masks'][:,:,indecies]

  result['rois'] = np.asarray(result['rois'])
  result['masks'] = np.asarray(result['masks'])
  result['class_ids'] = np.asarray(result['class_ids'])
  result['scores'] = np.asarray(result['scores'])
  return result

r = min_accuracy(r,0.8)
```
13. Display original image and the masked image.
```


### Download

To download the dataset images simply issue
```
python3 download.py
```
Alternatively, download from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3587843.svg)](https://doi.org/10.5281/zenodo.3587843)

Our API contains a jupyter notebook ``demo.pynb`` to inspect the dataset and visualize annotations.

**Unlabeled data**

A list of URLs for both unlabeled and labeled images is now also provided in `data/all_image_urls.csv`.
Each image contains one URL for each original image (second column) and one URL for a VGA-resized version (first column)
for images hosted by Flickr. If you decide to annotate these images using other tools, please make them public and contact us so we can keep track.

**Unofficial data**

Annotations submitted via our website are added weekly to `data/annotations_unofficial.json`. These have not yet been been reviewed by us -- some may be inaccurate or have poor segmentations. 
You can use the same command to download the respective images:
```
python3 download.py --dataset_path ./data/annotations_unofficial.json
```

### Trash Detection

The implementation of [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN)  is included in ``/detector``
with a few modifications. Requirements are the same. Before using this, the dataset needs to be split. You can either donwload our [weights and splits](https://github.com/pedropro/TACO/releases/tag/1.0) or generate these from scratch using the `split_dataset.py` script to generate 
N random train, val, test subsets. For example, run this inside the directory `detector`:
```
python3 split_dataset.py --dataset_dir ../data
```

For further usage instructions, check ``detector/detector.py``.

As you can see [here](http://tacodataset.org/stats), most of the original classes of TACO have very few annotations, therefore these must be either left out or merged together. Depending on the problem, ``detector/taco_config`` contains several class maps to target classes, which maintain the most dominant classes, e.g., Can, Bottles and Plastic bags. Feel free to make your own classes.

<p align="center">
<img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/teaser.gif" width="75%"/></p>
