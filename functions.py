# Imports
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
    # initialize result dictionary
    result = {'rois': [], 'masks': [], 'class_ids': [], 'scores': []}
    # list of indecies
    indecies = []
    # add the detected object info only if they have an accuracy greated than or equal to a
    for i,ele in enumerate(r['scores']):
        if ele >= a:
            result['rois'].append(r['rois'][i])
            result['class_ids'].append(r['class_ids'][i])
            result['scores'].append(r['scores'][i])
            indecies.append(i)
    # convert the lists within the result dictionary to numpy
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
        # handle the sample.JPG case
        # side note: we're running the sample.JPG through detection to lower the detection time of the first upload
        img_path = ''
        if img == 'sample.JPG':
            img_path = "static/assets/" + img
        else:
            img_path = "static/uploads/" + img
        
        # read the image
        image = skimage.io.imread(img_path)
        # initialize class names
        class_names = ["BG","Bottle","Bottle cap","Can","Cigarette","Cup","Lid","Other","Plastic bag + wrapper","Pop tab","Straw"]
        # run the image through detection using the loaded model
        r = model.detect([image], verbose=0)[0]
        # run image detection results through the min_accuracy function to get detected objects with accuracy greater than or equal to 90%
        r = min_accuracy(r,0.9)
        print('(',i,'/',len(images),') detected scores: ',r['scores'])
        # save the image with its corresponsing detected objects
        visualize.save_detected_img(image, img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        # add image info to json data file
        add_to_json(r,class_names,img)

def add_to_json(r,class_names,imageName):
    """
    Adds detected iamge and its corresponding info to data json file.
    
    :param r: list of results from the image detection
    :param class_names: list of all available trash category names
    :param imageName: name of the image that will be added to the json data file
    :return: 
    """
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
    """
    Removes a passed image.
    
    :param img: the image name to remove
    :return: 
    """
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

def allowed_file(filename):
    """
    Checks if the passed filename has an allowed extension.
    
    :param filename: filename to check the extension on
    :return: 
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in v.ALLOWED_EXTENSIONS

def get_images(images, offset=0, per_page=10):
    """
    Used to grab a set amount of the images for pagination in library page
    
    :param images: images to display
    :param offset: offset number (defaults to 0 if it isn't specified)
    :param per_page: number of images per page desired (defaults to 10 if it isn't specified)
    :return: list of images to display
    """
    return images[offset: offset + per_page]

def checkQuantity(quantityType, quantity):
    """
    Returning list of images that pass the quantity filter passed
    
    :param quantityType: the user input for the quantityType (greater than, less than, equal to, or empty if nothing is selected)
    :param quantity: The quantity the user have inputed
    :return: a list of the names of the images that pass the quantity filter
    """
    quantitySet = set()

    quantity = int(quantity)
    with open(v.JSON_DATA_FILE,'r') as json_data:
        # load the data
        data = json.load(json_data)

        for i in range(len(data["Images"])):
            # greater than
            if quantityType == 'Greater than':
                if data['Images'][i]['Quantity'] > quantity:
                    quantitySet.add(data['Images'][i]['Name'])
            # less than
            elif quantityType == 'Less than':
                if data['Images'][i]['Quantity'] < quantity:
                    quantitySet.add(data['Images'][i]['Name'])
            # equal to
            elif quantityType == 'Equal to':
                if data['Images'][i]['Quantity'] == quantity:
                    quantitySet.add(data['Images'][i]['Name'])

    return quantitySet

def checkClasses(selected_trash, intersection):
    """
    Returns list of images that has at least one of the selected trash categories if the intersection
    is not selected and images that has all of the selected trash categories if intersection checkbox
    is selected.
    
    :param selected_trash: list of the selected trash categories
    :param intersection: True if intersection checkbox is selected and False otherwise
    :return: list of the names of the images that has at least one of the selected trash categories
    """
    classSet = set()

    with open(v.JSON_DATA_FILE,'r') as json_data:
        # loads data
        data = json.load(json_data)
        # if intersection checkbox is not selected
        if intersection == "":
            for i in selected_trash:
                for j in range(len(data["Images"])):
                    if i in data['Images'][j]['Classes']:
                        classSet.add(data['Images'][j]['Name'])
        # if intersection checkbox is selected
        else:
            for i in range(len(data["Images"])):
                if set(selected_trash) <= set(data['Images'][i]['Classes']):
                    classSet.add(data['Images'][i]['Name'])

    return classSet

def get_classes_info():
    """
    Returns a dictionary with each image and its trash categories and corresponding count.
    
    :return: a dictionary with each image and its trash categories and corresponding count
    """
    class_info = dict()
    with open(v.JSON_DATA_FILE,'r') as json_data:
        # loads data
        data = json.load(json_data)
        # use imported Counter package to get the number of each trash class category within each image
        for img in data["Images"]:
            c = Counter(img["Classes"])
            class_info[img["Name"]] = dict(c)
    return class_info
