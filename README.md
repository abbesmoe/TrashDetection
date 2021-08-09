<h1 align="center">Trash Detection</h1>

This is a web app implementation of the TACO dataset and Mask R-CNN on Python 3, Keras, and TensorFlow. The model generates an instance segmentation with masks, bounding boxes, class names, and accuracy scores on an image. <br>

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

# This Repository Includes:
* Annotated image data in ```/data```
* Implementation of Mask-RCNN in ```/detector```
* Colab notebooks to train and test the model in ```/samples```
* Uploaded and annotated images in ```/static```
* Website html code in ```/templates```

**Implemented Repositories:**
* Taco dataset: https://github.com/pedropro/TACO
* Mask-RCNN: https://github.com/matterport/Mask_RCNN

# Improvements
* Updated tensorflow and keras to the newest versions
* Added more images to the dataset and re-trained the model for better accuracy

# Suggested Improvements
* Train over the dataset more
* Research how hyperparameter tuning can help increase the accuracy of the model
* Add more images to the dataset
* Optimize the model for better runtime
* Add a location feature

# Getting started

### Requirements
* python 3.7
* tensorflow 2.5
* keras 2.5

### How to setup
1. Open Google CoLab and create a new notebook
2. Clone the repository 
```
!git clone https://github.com/abbesmoe/TrashDetection
```
3. Install the requred packages
```
!pip install keras==2.5.0rc0
!pip install tensorflow==2.5
!pip install 'h5py==2.10.0'
```
4. Change the directory 
```
%cd TrashDetection/
```
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
7. Create config class and override hyperparamters
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
imageName = "Your_image.jpg"
img = Image.open(os.path.join(imagePath, imageName))

fileName = os.path.splitext(imageName)[0]
fileExtension = os.path.splitext(imageName)[1]

if fileExtension != '.jpg':
  img.convert('RGB').save("{}{}.jpg".format(imagePath, fileName))
  imageName = "{}.jpg".format(fileName)

image = skimage.io.imread(os.path.join(imagePath, imageName))
```
11. Run the detection on an image, skimage helps with image processing on a computer.
```
def detection(images):
    global model
    import skimage.io
   
    for img in images:
        print(img)
        img_path = "static/uploads/" + img
        image = skimage.io.imread(img_path)

r = model.detect([image], verbose=0)[0]
```
12. Define the minimum accuracy displayed after detection.
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
13. Display original image while also displaying the masked image.
```
plt.figure(figsize=(12,10))
skimage.io.imshow(image)

visualize.display_instances(image, imageName, r['rois'], r['masks'],
                            r['class_ids'], class_names, r['scores'])
```
14. Write an empty json file
```
# run only once, unless you want to empty data.json
with open('data.json', 'w') as f:
  data = {"Images":[]}
```
15. Add data from detected image to data.json.
```
size = len(r['class_ids'])
i = 0
classNameList = []
while i < size:
  obj_name = class_names[r['class_ids'][i]]
  classNameList.append(obj_name)
  i = i + 1

def write_json(data, filename="data.json"):
  with open(filename, "w") as f:
    json.dump(data, f, indent=4)


img_data = {}
imgdata["Name"] = "annotated{}".format(imageName)
img_data["Quantity"] = len(classNameList)
img_data['Classes'] = classNameList

data['Images'].append(img_data)

write_json(data)
```
16. Display a dataframe from data.json.
```
pd.set_option("display.max_rows", None, "display.max_columns", None)

with open('data.json') as json_data:
    data = json.load(json_data)

df = pd.DataFrame(data['Images'])
print(df)
```
17. Checks for selected quantity of specific trash types in an image and loops through them to verify the conditions and output an image.


  
        def checkQuantity():
        inp = int(input("Select a quantity: "))
        userSign = "Greater than"

        with open("../../data.json",'r') as json_data:
        data = json.load(json_data)
        for i in range(len(data["Images"])):
        if userSign == 'Greater than':
        if data['Images'][i]['Quantity'] >= inp:
          print(data['Images'][i]['Name'])

        elif userSign == 'Less than':
        if data['Images'][i]['Quantity'] <= inp:
          print(data['Images'][i]['Name'])

        elif userSign == 'Equal to':
        if data['Images'][i]['Quantity'] == inp:
          print(data['Images'][i]['Name'])

        else:
        print(data['Images'][i]['Name'])
         checkQuantity()

### Download

To download the original 1500 taco dataset images we started with, simply issue:
```
%cd TrashDetection/
!python3 download.py
```

### The Use of AWS

We used an AWS ec2 instance to train our model. We struggled with utilitizing the GPU on the instances and found out our initial AMI did not come with graphics card drivers (NVIDIA drivers). We mainly tried two different instances:
* p3.2xlarge is cheaper and trained one epoch in roughly 16 minutes.
* p3.16xlarge was eight times the price but was not training eight times faster, training one epoch in roughly 14 minutes.
Thus we stuck with the p3.2xlarge instance.

**How to Link s3 bucket to ec2 instance**

From the ec2 instance command line, run:
```
aws configure
```
For the required information: use the access keys from account that has the S3 bucket. You can find them under Account Security Credentials. For more information, [click here](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).

Once access keys are entered, the s3 bucket should be visible with this command:
```
aws s3 ls
```

[This](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AmazonS3.html) is how we copied data to and from the instance.
