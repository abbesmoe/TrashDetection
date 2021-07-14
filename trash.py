# Imports
from flask import Flask, redirect, url_for, render_template, request, flash, send_file
import os
from werkzeug.utils import secure_filename
from flask_paginate import Pagination, get_page_args
import json
import threading

# Starts the web app
app = Flask(__name__)

# Setting global variables
trash_list = ["Bottle", "Pop tab", "Can", "Bottle cap", "Cigarette", "Cup", "Lid", "Other", "Plastic bag + wrapper", "Straw"]
selected_trash_list = []
images_data = {"Images":[]}
headings = ["Images","Quantity"]
data = []
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['jpg','png','jpeg','img','gif','mp4'])

type = ""
quantity = ""
quantityType = ""
intersection = "False"

# Used to grab a set amount of the images for pagination in library page
def get_images(images, offset=0, per_page=10):
    return images[offset: offset + per_page]

# Loads the model
def load_model():
    ## Load Dataset
    import csv
    import detector.dataset as dataset

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

    from detector.config import Config
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

    import detector.model as modellib
    # Create model objects in inference mode.
    # inference mode means we are taking live data points to calculate an output #
    # model dir is where the trained model is saved #
    # config: exposes a config class, using those settings #
    model = modellib.MaskRCNN(mode="inference", model_dir='mask_rcnn_coco.hy', config=config)

    # Load weights trained on MS-COCO
    # The trained objects that are used for accuracy #
    # by_name: weights are loaded into layers only if the share the same name #, exclude=["mrcnn_bbox_fc","mrcnn_class_logits","mrcnn_mask"]
    model.load_weights('mask_rcnn_taco_0100.h5', by_name=True, weights_out_path=None)
    model.keras_model._make_predict_function()
    return model

# Loads the model
model = load_model()

# Run detection on an image
def detection(img):
    global model
    import skimage.io
    # load an image #
    #skimage helps with image processing on a computer #
    img_path = "static/uploads/" + img
    image = skimage.io.imread(img_path)

    # Remove alpha channel, if it has one
    if image.shape[-1] == 4:
        image = image[..., :3]

    class_names = ["BG","Bottle","Bottle cap","Can","Cigarette","Cup","Lid","Other","Plastic bag + wrapper","Pop tab","Straw"]

    r = model.detect([image], verbose=0)[0]

    print(r['scores'])

    import detector.visualize as visualize
    visualize.display_instances(image, img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

    add_to_json(r,class_names,img)

    return

# Add image information to a json file after running through detection
def add_to_json(r,class_names,imageName):
    global images_data

    classNameList = []
    for i in range(len(r['class_ids'])):
        obj_name = class_names[r['class_ids'][i]]
        classNameList.append(obj_name)

    def write_json(data, filename="data.json"):
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


    img_data = {}
    img_data["Name"] = "annotated_{}".format(imageName)
    img_data["Quantity"] = len(classNameList)
    img_data['Classes'] = classNameList

    images_data['Images'].append(img_data)

    write_json(images_data)

# Checks if the uploaded images have supported extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Download an Image from the library page
@app.route('/download', methods=['GET'])
def download_file():
    is_ann = request.args.get("is_ann")
    if is_ann == "True":
        image = "static/annotated_images/"+request.args.get("img")
        return send_file(image,as_attachment=True)
    else:
        image = "static/uploads/"+request.args.get("img")
        return send_file(image,as_attachment=True)

# Removes an Image from the library page
@app.route('/remove', methods=['GET'])
def remove_file():
    img = request.args.get("img")
    uploaded_img = "static/uploads/"+img
    ann_img = "static/annotated_images/output_"+img
    os.remove(uploaded_img)
    os.remove(ann_img)
    return redirect(url_for("library"))

# Library Page
@app.route("/library")
def library():
    path = "static/uploads"
    images = os.listdir(path)

    path2 = "static/annotated_images"
    ann_images = os.listdir(path2)

    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    total = len(images)
    pagination_images = get_images(images, offset=offset, per_page=per_page)
    pagination_ann_images = get_images(ann_images, offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')
    return render_template("library.html", 
                           images=pagination_images,
                           ann_images=pagination_ann_images,
                           list_len=len(images),
                           page=page,
                           per_page=per_page,
                           pagination=pagination)

# Redirects to the upload page
@app.route("/uploadredirect")
def uploadredirect():
    return redirect(url_for("upload"))

# Upload Page
@app.route("/upload")
def upload():
    return render_template("upload.html")

# Upload Page when a post method is envoked
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'files[]' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('files[]')
    if files[0].filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
    for file in files:
        if not (file and allowed_file(file.filename)):
            flash('Allowed image types are - png, jpg, jpeg, gif, img, tif, tiff, bmp, eps, raw, mp4, mov, wmv, flv, avi')
            return redirect(request.url)
    flash('You can view all your uploaded files in the library page')
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print('upload_image filename: ' + filename)
            flash(filename + ' has been successfully uploaded')

            threading.Thread(target=detection(filename)).start()
    ann_image= "output_" + filename

    return render_template('upload.html', filename=filename, ann_image=ann_image)

# Displays an Image
@app.route('/display/<filename>')
def display_image(filename, is_ann=False):
    is_ann = request.args.get("is_ann")
    if is_ann == "True":
        return redirect(url_for('static', filename='annotated_images/' + filename))
    else:
        return redirect(url_for('static', filename='uploads/' + filename))

# Redirects to the search page
@app.route("/searchredirect")
def searchredirect():
    return redirect(url_for("search"))

def checkQuantity(quantityType, quantity):
    quantitySet = set()

    quantity = int(quantity)
    with open("data.json",'r') as json_data:
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

    with open("data.json",'r') as json_data:
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

# Search Page
@app.route("/search", methods=["POST", "GET"])
def search():
    global selected_trash_list
    global trash_list
    global headings
    global data
    global type
    global quantity
    global quantityType
    global intersection
    result = render_template("search.html", trash_list=trash_list, selected_trash_list=selected_trash_list, headings=headings, data=data, style="none", type=type, quantity=quantity, quantityType=quantityType, intersection = intersection)
    if request.method == "POST":
        if "+" in request.form:
            if "trash" in request.form:
                trash = request.form["trash"]
                trash_list.remove(trash)
                selected_trash_list.append(trash)
                result = render_template("search.html", trash_list=trash_list, selected_trash_list=selected_trash_list, headings=headings, data=data, style="none", type=type, quantity=quantity, quantityType=quantityType, intersection = intersection)
        elif "-" in request.form:
            if "selectedtrash" in request.form:
                selectedtrash = request.form["selectedtrash"]
                selected_trash_list.remove(selectedtrash)
                trash_list.append(selectedtrash)
                result = render_template("search.html", trash_list=trash_list, selected_trash_list=selected_trash_list, headings=headings, data=data, style="none", type=type, quantity=quantity, quantityType=quantityType, intersection = intersection)
        elif "Search" in request.form:
            data = []
            headings = ["Images", "Quantity"]
            with open("data.json",'r') as json_data:
                imgs_data = json.load(json_data)
                # recyclables or nonrecyclables
                if "type" in request.form:
                    type = request.form["type"]
                    if type != "":
                        headings.append(type)
                # adds selected trash to the headings
                for t in selected_trash_list:
                    headings.append(t)

                if "quantity" in request.form:
                    quantity = request.form["quantity"]
                if "quantityType" in request.form:
                    quantityType = request.form["quantityType"]
                #handle quantity
                quantitySet = set()
                if quantity != "" and quantityType != "":
                    quantitySet = checkQuantity(quantityType, quantity)
                
                intersect = ""
                if "Intersection" in request.form:
                    intersect = request.form["Intersection"]
                    intersection = "True"
                else:
                    intersection = "False"

                classSet = checkClasses(selected_trash_list, intersect)

                finalSet = set()
                if len(quantitySet) == 0 and len(classSet) == 0 and (quantity != "" or quantityType != "" or len(selected_trash_list) != 0):
                    #print error
                    #print nothing
                    finalSet = set()
                elif len(quantitySet) == 0 and len(classSet) == 0:
                    for image in imgs_data["Images"]:
                        finalSet.add(image["Name"])
                elif len(quantitySet) == 0 and len(classSet) != 0 and (quantity != "" or quantityType != ""):
                    finalSet = set()
                elif len(quantitySet) == 0 and len(classSet) != 0:
                    finalSet = classSet
                elif len(quantitySet) != 0 and len(classSet) == 0 and len(selected_trash_list) != 0:
                    finalSet = set()
                elif len(quantitySet) != 0 and len(classSet) == 0:
                    finalSet = quantitySet
                else:
                    finalSet = quantitySet.intersection(classSet)

                for image in imgs_data["Images"]:
                    img_data = []

                    # img_data.append(image["Name"])
                    # img_data.append(image["Quantity"])

                    for n in finalSet:
                        if image["Name"] == n:
                            img_data.append(image["Name"])
                            img_data.append(image["Quantity"])

                    data.append(img_data)
            result = render_template("search.html", trash_list=trash_list, selected_trash_list=selected_trash_list, headings=headings, data=data, style="inline", type=type, quantity=quantity, quantityType=quantityType, intersection = intersection)
    return result

# Runs the web app
if __name__ == "__main__":
    app.run(debug=True)
