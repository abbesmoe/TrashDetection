import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__)

## Load Dataset
import csv
import dataset

# Load class map - these tables map the original TACO classes to your desired class system
# and allow you to discard classes that you don't want to include.
class_map = {}
with open("detector/taco_config/map_10.csv") as csvfile:
    reader = csv.reader(csvfile)
    class_map = {row[0]:row[1] for row in reader}

# Load full dataset or a subset
TACO_DIR = "data"
round = None # Split number: If None, loads full dataset else if int > 0 selects split no 
subset = "train" # Used only when round !=None, Options: ('train','val','test') to select respective subset
dataset = dataset.Taco()
taco = dataset.load_taco(TACO_DIR, round, subset, class_map=class_map, return_taco=True)

# Must call before using the dataset
dataset.prepare()

print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

from config import Config
class TacoTestConfig(Config):
    NAME = "taco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.3
    NUM_CLASSES = dataset.num_classes
#     IMAGE_MAX_DIM = 1024
#     IMAGE_MIN_DIM = 1024
#     IMAGE_RESIZE_MODE = "square"
config = TacoTestConfig()
config.display()

import model as modellib
# Create model objects in inference mode.
# inference mode means we are taking live data points to calculate an output #
# model dir is where the trained model is saved #
# config: exposes a config class, using those settings #
model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

# Load weights trained on MS-COCO
# The trained objects that are used for accuracy #
# by_name: weights are loaded into layers only if the share the same name #, exclude=["mrcnn_bbox_fc","mrcnn_class_logits","mrcnn_mask"]
model.load_weights('mask_rcnn_taco_0100.h5', by_name=True, weights_out_path=None)

import skimage.io
# load an image #
#skimage helps with image processing on a computer #
image = skimage.io.imread('1.jpg')

class_names = ["BG","Bottle","Bottle cap","Can","Cigarette","Cup","Lid","Other","Plastic bag + wrapper","Pop tab","Straw"]

r = model.detect([image], verbose=0)[0]

print(r['scores'])

import visualize
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])