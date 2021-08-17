from detector.config import Config
from collections import Counter
import csv
import detector.dataset as datasets
import detector.model as modellib
import skimage.io
import detector.visualize as visualize
import variables as v
import os
import json
import numpy as np

def load_model():
    """
    Loads the model when app starts.
    
    :return: returns the model which then will be used for detection
    """
    # Load class map - these tables map the original TACO classes to your desired class system
    # and allow you to discard classes that you don't want to include.
    class_map = {}
    with open(v.MAP_FILE) as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]:row[1] for row in reader}

    # Load full dataset or a subset
    round = None                # Split number: If None, loads full dataset else if int > 0 selects split no 
    subset = "train"            # Used only when round !=None, Options: ('train','val','test') to select respective subset
    dataset = datasets.Taco()
    dataset.load_taco(v.TACO_DIR, round, subset, class_map=class_map, return_taco=False)

    # Must call before using the dataset
    dataset.prepare()

    # Print the classes
    print("\n---------- Classes ----------\n")
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    # Set the config
    class TacoTestConfig(Config):
        NAME = "taco"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.3
        NUM_CLASSES = dataset.num_classes
    config = TacoTestConfig()
    
    # print configurations
    print("\n---------- CONFIG HYPERPARAMETERS ----------")
    config.display()

    # Create model objects in inference mode.
    # inference mode means we are taking live data points to calculate an output #
    # model dir is where the trained model is saved #
    # config: exposes a config class, using those settings #
    model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

    # Load weights trained on MS-COCO
    # The trained objects that are used for accuracy #
    # by_name: weights are loaded into layers only if the share the same name #, exclude=["mrcnn_bbox_fc","mrcnn_class_logits","mrcnn_mask"]
    model.load_weights('weights/mask_rcnn_taco_0100.h5', by_name=True, weights_out_path=None)
    model.keras_model.make_predict_function()
    return model

def min_accuracy(r,a):
    """
    Filter the list of detected objects to only contain objects with accuracy grater than parameter a.
    
    :param r: list of results from the image detection
    :param a: the minimum accuracy value
    :return: list of results from the image detection with object that have accuracy grater than parameter a
    """
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

def detection(model, images):
    """
    Perform detection using the model passed on the list of images.
    
    :param model: model which was loaded when the web app started to perfect detection
    :param images: list of uploaded images that will run through detection
    :return:
    """
    for i, img in enumerate(images):
        img_path = ''
        if img == 'sample.JPG':
            img_path = "static/assets/" + img
        else:
            img_path = "static/uploads/" + img
        image = skimage.io.imread(img_path)
        class_names = ["BG","Bottle","Bottle cap","Can","Cigarette","Cup","Lid","Other","Plastic bag + wrapper","Pop tab","Straw"]
        r = model.detect([image], verbose=0)[0]
        r = min_accuracy(r,0.9)
        print('(',i,'/',len(images),') detected scores: ',r['scores'])
        visualize.display_instances(image, img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        add_to_json(r,class_names,img)

def add_to_json(r,class_names,imageName):
    """
    Adds detected iamge and its corresponding info to data json file.
    
    :param r: list of results from the image detection
    :param class_names: list of all available trash category names
    :param imageName: name of the image that will be added to the json data file
    :return: 
    """
    global images_data
    classNameList = []
    for i in range(len(r['class_ids'])):
        obj_name = class_names[r['class_ids'][i]]
        classNameList.append(obj_name)
    img_data = {}
    img_data["Name"] = imageName
    img_data["Quantity"] = len(classNameList)
    img_data['Classes'] = classNameList
    v.IMAGES_DATA['Images'].append(img_data)
    with open(v.JSON_DATA_FILE, "w") as f:
        json.dump(v.IMAGES_DATA, f, indent=4)

def remove(img):
    uploaded_img = v.UPLOAD_PATH+img
    ann_img = v.ANNOTATED_IMAGES_PATH + "output_" +img
    print('annotated image: ',ann_img)
    if img == v.SAMPLE_IMG:
        os.remove(ann_img)
    else:
        os.remove(ann_img)
        os.remove(uploaded_img)
    ann_name = img
    for i, img in enumerate(v.IMAGES_DATA['Images']):
        if img['Name'] == ann_name:
            v.IMAGES_DATA['Images'].pop(i)
    with open(v.JSON_DATA_FILE, "w") as f:
        json.dump(v.IMAGES_DATA, f, indent=4)

# Checks if the uploaded images have supported extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in v.ALLOWED_EXTENSIONS

# Used to grab a set amount of the images for pagination in library page
def get_images(images, offset=0, per_page=10):
    return images[offset: offset + per_page]

def checkQuantity(quantityType, quantity):
    quantitySet = set()

    quantity = int(quantity)
    with open(v.JSON_DATA_FILE,'r') as json_data:
        data = json.load(json_data)

        for i in range(len(data["Images"])):
            if quantityType == 'Greater than':
                if data['Images'][i]['Quantity'] > quantity:
                    quantitySet.add(data['Images'][i]['Name'])

            elif quantityType == 'Less than':
                if data['Images'][i]['Quantity'] < quantity:
                    quantitySet.add(data['Images'][i]['Name'])

            elif quantityType == 'Equal to':
                if data['Images'][i]['Quantity'] == quantity:
                    quantitySet.add(data['Images'][i]['Name'])

    return quantitySet

def checkClasses(selected_trash, intersection):
    # REPLACE with get method from search page
    classSet = set()

    with open(v.JSON_DATA_FILE,'r') as json_data:
        data = json.load(json_data)
        if intersection == "":
            for i in selected_trash:
                for j in range(len(data["Images"])):
                    if i in data['Images'][j]['Classes']:
                        classSet.add(data['Images'][j]['Name'])
        else:
            for i in range(len(data["Images"])):
                if set(selected_trash) <= set(data['Images'][i]['Classes']):
                    classSet.add(data['Images'][i]['Name'])

    return classSet

def get_classes_info():
    class_info = dict()
    with open(v.JSON_DATA_FILE,'r') as json_data:
        data = json.load(json_data)
        for img in data["Images"]:
            c = Counter(img["Classes"])
            class_info[img["Name"]] = dict(c)
    return class_info
