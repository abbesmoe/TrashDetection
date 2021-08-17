# Imports
from flask import Flask, redirect, url_for, render_template, request, flash, send_file, jsonify, make_response
import os
from werkzeug.utils import secure_filename
from flask_paginate import Pagination, get_page_args
import json
import threading
import zipfile
import variables as v
import functions as func

# Starts the web app
app = Flask(__name__)

# app configurations
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = v.UPLOAD_PATH
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Dictionary for the json file to store image data
with open(v.JSON_DATA_FILE,'r') as json_data:
    v.IMAGES_DATA = json.load(json_data)

# Loading Sample Image
func.detection(v.MODEL,[v.SAMPLE_IMG])
ann_images = os.listdir(v.ANNOTATED_IMAGES_PATH)
if v.SAMPLE_ANN_IMG in ann_images:
    func.remove(v.SAMPLE_IMG)

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

# Download all Images from the library page and save it in a zip file
@app.route('/downloadall', methods=['GET'])
def download_files():
    zipf = zipfile.ZipFile('Images.zip','w', zipfile.ZIP_DEFLATED)
    is_ann = request.args.get("is_ann")
    images = request.args.getlist("images")
    for image in images:
        if is_ann == "True":
            image_path = "static/annotated_images/"+image
        else:
            image_path = "static/uploads/"+image
        zipf.write(image_path)
    zipf.close()
    return send_file('Images.zip', mimetype = 'zip', attachment_filename= 'Images.zip' ,as_attachment=True)

# Removes one Image from the library page
@app.route('/remove', methods=['GET'])
def remove_file():
    img = request.args.get("img")
    func.remove(img)
    return redirect(url_for("library"))

# Removes all Images from the library page
@app.route('/removeall', methods=['GET'])
def remove_files():
    images = request.args.getlist("images")
    v.IMAGES_DATA = {"Images":[]}
    func.write_json(v.IMAGES_DATA)
    for image in images:
        uploaded_img = "static/uploads/"+image
        ann_img = "static/annotated_images/output_"+image
        os.remove(uploaded_img)
        os.remove(ann_img)
    return redirect(url_for("library"))

# Library Page
@app.route("/library")
def library():
    images = os.listdir(v.UPLOAD_PATH)
    ann_images = os.listdir(v.ANNOTATED_IMAGES_PATH)

    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    total = len(images)
    # pagination allows us to break up the page into multiple pages. We set the number of images displayed per page to 10 and anything more will be put on a new page
    pagination_images = func.get_images(images, offset=offset, per_page=per_page)
    pagination_ann_images = func.get_images(ann_images, offset=offset, per_page=per_page)
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')
    return render_template("library.html", 
                           images=pagination_images,
                           ann_images=pagination_ann_images,
                           page=page,
                           per_page=per_page,
                           pagination=pagination)

# Upload Page when a post method is envoked
@app.route('/upload', methods=['POST','GET'])
def upload():
    if request.method == "POST":
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        if files[0].filename == '':
                flash('No image selected for uploading')
                return redirect(request.url)
        for file in files:
            if not (file and func.allowed_file(file.filename)):
                flash('Allowed image types are - png, jpg, jpeg, gif, img, tif, tiff, bmp, eps, raw, mp4, mov, wmv, flv, avi')
                return redirect(request.url)
        flash('You can view all your uploaded files in the library page')
        # thread = ""
        images= []
        for file in files:
            if file and func.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                #print('upload_image filename: ' + filename)
                flash(filename + ' has been successfully uploaded')
                
                images.append(filename)
        thread = threading.Thread(target=func.detection,args=[v.MODEL, images])
        thread.start()
        thread.join()

    return render_template('upload.html')

# Displays an Image
@app.route('/display/<filename>')
def display_image(filename, is_ann=False):
    is_ann = request.args.get("is_ann")
    if is_ann == "True":
        return redirect(url_for('static', filename='annotated_images/' + filename))
    else:
        return redirect(url_for('static', filename='uploads/' + filename))

# Search Page
@app.route("/search", methods=["POST", "GET"])
def search():
    result = render_template("search.html", trash_list=v.TRASH_LIST, selected_trash_list=v.SELECTED_TRASH_LIST, headings=v.SEARCH_TABLE_HEADERS, data=v.SEARCH_TABLE_ROWS, style="none", recyclable=v.RECYCLABLE_FILTER, non_recyclable=v.NON_RECYCLABLE_FILTER, quantity=v.QUANTITY_FILTER, quantityType=v.QUANTITY_TYPE_FILTER, intersection = v.INTERSECTION_FILTER)
    if request.method == "POST":
        # this code displays how an item is added to the search list
        if "+" in request.form:
            if "trash" in request.form:
                trash = request.form["trash"]
                v.TRASH_LIST.remove(trash)
                v.SELECTED_TRASH_LIST.append(trash)
                result = render_template("search.html", trash_list=v.TRASH_LIST, selected_trash_list=v.SELECTED_TRASH_LIST, headings=v.SEARCH_TABLE_HEADERS, data=v.SEARCH_TABLE_ROWS, style="none", recyclable=v.RECYCLABLE_FILTER, non_recyclable=v.NON_RECYCLABLE_FILTER, quantity=v.QUANTITY_FILTER, quantityType=v.QUANTITY_TYPE_FILTER, intersection = v.INTERSECTION_FILTER)
      # this code displays how an item is removed from the search list
        elif "-" in request.form:
            if "selectedtrash" in request.form:
                selectedtrash = request.form["selectedtrash"]
                v.SELECTED_TRASH_LIST.remove(selectedtrash)
                v.TRASH_LIST.append(selectedtrash)
                result = render_template("search.html", trash_list=v.TRASH_LIST, selected_trash_list=v.SELECTED_TRASH_LIST, headings=v.SEARCH_TABLE_HEADERS, data=v.SEARCH_TABLE_ROWS, style="none", recyclable=v.RECYCLABLE_FILTER, non_recyclable=v.NON_RECYCLABLE_FILTER, quantity=v.QUANTITY_FILTER, quantityType=v.QUANTITY_TYPE_FILTER, intersection = v.INTERSECTION_FILTER)
        elif "Search" in request.form:
            v.SEARCH_TABLE_HEADERS = ["Images", "Quantity"]
            v.SEARCH_TABLE_ROWS = []
            with open(v.JSON_DATA_FILE,'r') as json_data:
                imgs_data = json.load(json_data)
                # recyclables or nonrecyclables
                # check if intersection checkbox is checked
                if "Recyclables" in request.form:
                    r = request.form["Recyclables"]
                    v.SEARCH_TABLE_HEADERS.append(r)
                    v.RECYCLABLE_FILTER = "True"
                else:
                    v.NON_RECYCLABLE_FILTER = "False"
                if "Non_recyclables" in request.form:
                    non_r = request.form["Non_recyclables"]
                    v.SEARCH_TABLE_HEADERS.append(non_r)
                    v.NON_RECYCLABLE_FILTER = "True"
                else:
                    v.NON_RECYCLABLE_FILTER = "False"

                # adds selected trash to the headings
                for t in v.SELECTED_TRASH_LIST:
                    v.SEARCH_TABLE_HEADERS.append(t)

                if "quantity" in request.form:
                    quantity = request.form["quantity"]
                if "quantityType" in request.form:
                    quantityType = request.form["quantityType"]
                #handle quantity
                quantitySet = set()
                if quantity != "" and quantityType != "":
                    quantitySet = func.checkQuantity(quantityType, quantity)
                
                # check if intersection checkbox is checked
                intersect = ""
                if "Intersection" in request.form:
                    intersect = request.form["Intersection"]
                    v.INTERSECTION_FILTER = "True"
                else:
                    v.INTERSECTION_FILTER = "False"

                classSet = func.checkClasses(v.SELECTED_TRASH_LIST, intersect)

                finalSet = set()
                if len(quantitySet) == 0 and len(classSet) == 0 and (quantity != "" or quantityType != "" or len(v.SELECTED_TRASH_LIST) != 0):
                    #print error
                    #print nothing
                    finalSet = set()
                    # this checks through the different specifcations the user seletec to searh for and "finalSet" will print the images that match the selected search options
                elif len(quantitySet) == 0 and len(classSet) == 0:
                    for image in imgs_data["Images"]:
                        finalSet.add(image["Name"])
                elif len(quantitySet) == 0 and len(classSet) != 0 and (quantity != "" or quantityType != ""):
                    finalSet = set()
                elif len(quantitySet) == 0 and len(classSet) != 0:
                    finalSet = classSet
                elif len(quantitySet) != 0 and len(classSet) == 0 and len(v.SELECTED_TRASH_LIST) != 0:
                    finalSet = set()
                elif len(quantitySet) != 0 and len(classSet) == 0:
                    finalSet = quantitySet
                else:
                    finalSet = quantitySet.intersection(classSet)
                
                print(finalSet)

                for image in imgs_data["Images"]:
                    img_data = []
                    classes_info = func.get_classes_info()
                    for n in finalSet:
                        if image["Name"] == n:
                            img_data.append(image["Name"])
                            img_data.append(image["Quantity"])
                            r_count = 0
                            # counts the number of recyclable items in the image
                            if v.RECYCLABLE_FILTER == "True":
                                for c,c_count in classes_info[n].items():
                                    if c in v.RECYCLABLES:
                                        r_count+=c_count
                                img_data.append(r_count)
                            nonr_count = 0
                             # counts the number of non-recyclable items in the image
                            if v.NON_RECYCLABLE_FILTER == "True":
                                for c,c_count in classes_info[n].items():
                                    if c in v.NON_RECYCLABLES:
                                        nonr_count+=c_count
                                img_data.append(nonr_count)

                            for t in v.SELECTED_TRASH_LIST:
                                if t in classes_info[n]:
                                    img_data.append(classes_info[n][t])
                                else:
                                    img_data.append(0)

                    v.SEARCH_TABLE_ROWS.append(img_data)
            result = render_template("search.html", trash_list=v.TRASH_LIST, selected_trash_list=v.SELECTED_TRASH_LIST, headings=v.SEARCH_TABLE_HEADERS, data=v.SEARCH_TABLE_ROWS, style="inline", recyclable=v.RECYCLABLE_FILTER, non_recyclable=v.NON_RECYCLABLE_FILTER, quantity=v.QUANTITY_FILTER, quantityType=v.QUANTITY_TYPE_FILTER, intersection = v.INTERSECTION_FILTER)
    return result

# Runs the web app
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.3', port=5000)
