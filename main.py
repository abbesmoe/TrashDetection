# Imports
from flask import Flask, redirect, url_for, render_template, request, flash, send_file, jsonify, make_response
from werkzeug.utils import secure_filename
from flask_paginate import Pagination, get_page_args
import os
import json
import threading
import zipfile
import variables as v
import functions as func

# App configurations
app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = v.UPLOAD_PATH
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load existing json image saved data
with open(v.JSON_DATA_FILE,'r') as json_data:
    v.IMAGES_DATA = json.load(json_data)

# Loading Sample Image
# Gets the detection to run when the app starts which
# results in decreasing detection runtime for the first upload
func.detection(v.MODEL,[v.SAMPLE_IMG])
ann_images = os.listdir(v.ANNOTATED_IMAGES_PATH)
if v.SAMPLE_ANN_IMG in ann_images:
    func.remove(v.SAMPLE_IMG)

@app.route("/")
def home():
    """
    Home page.
    
    :return: returns the index.html template
    """
    return render_template("index.html")

@app.route('/upload', methods=['POST','GET'])
def upload():
    """
    Upload Page. Uploads all selected files and pass them through the model for detection.
    
    :return: returns the upload.html template
    """
    # if "Upload and Detect" button is pressed
    if request.method == "POST":
        # check if any files have been selected
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        # check if at least one files has been selected
        if files[0].filename == '':
                flash('No image selected for uploading')
                return redirect(request.url)
        # check if the selected files are of allowed types
        for file in files:
            if not (file and func.allowed_file(file.filename)):
                flash('Allowed image types are - png, jpg, jpeg, gif, img, tif, tiff, bmp, eps, raw, mp4, mov, wmv, flv, avi')
                return redirect(request.url)
        flash('You can view all your uploaded files in the library page')
        images= []
        # upload each of the selected files
        for file in files:
            if file and func.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                #print('upload_image filename: ' + filename)
                flash(filename + ' has been successfully uploaded')
                
                images.append(filename)
        # run detection
        # used threading to run the detection so when the user moves to other tabs in the web app the detection will still continue to run
        thread = threading.Thread(target=func.detection,args=[v.MODEL, images])
        thread.start()
        thread.join()

    return render_template('upload.html')

@app.route("/library")
def library():
    """
    Library page.
    
    :return: returns the library.html template
    """
    # get all images and annotated images
    images = os.listdir(v.UPLOAD_PATH)
    ann_images = os.listdir(v.ANNOTATED_IMAGES_PATH)

    
    # used pagination to allows us to break up the page into multiple pages
    # set the number of images displayed per page to 10 and anything more will be put on a new page
    total = len(images)
    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
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

@app.route('/download', methods=['GET'])
def download_file():
    """
    Downloads file (used in library and search page tables).
    
    :return: send_file function which downloads the file
    """
    # check if the passed file is the original image or the annotated image
    is_ann = request.args.get("is_ann")
    # annotated image
    if is_ann == "True":
        image = "static/annotated_images/"+request.args.get("img")
        return send_file(image,as_attachment=True)
    # original image
    else:
        image = "static/uploads/"+request.args.get("img")
        return send_file(image,as_attachment=True)

@app.route('/downloadall', methods=['GET'])
def download_files():
    """
    Downloads all files (used in library page).
    
    :return: send_file function which downloads all the files in a zip folder
    """
    # create a zip file
    zipf = zipfile.ZipFile('Images.zip','w', zipfile.ZIP_DEFLATED)
    # check if the passed files are the original images or the annotated images
    is_ann = request.args.get("is_ann")
    # all the images to download
    images = request.args.getlist("images")
    for image in images:
        # annotated images
        if is_ann == "True":
            image_path = "static/annotated_images/"+image
        # original image
        else:
            image_path = "static/uploads/"+image
        # add the images to the zip file
        zipf.write(image_path)
    zipf.close()
    return send_file('Images.zip', mimetype = 'zip', attachment_filename= 'Images.zip' ,as_attachment=True)

@app.route('/remove', methods=['GET'])
def remove_file():
    """
    Removes file (used in library page).
    
    :return: redircts back to the library page
    """
    # image to remove
    img = request.args.get("img")
    # remove the image
    func.remove(img)
    return redirect(url_for("library"))

@app.route('/removeall', methods=['GET'])
def remove_files():
    """
    Removes all files (used in library page).
    
    :return: redirects back to the library page
    """
    # all images to remove
    images = request.args.getlist("images")
    # reset all json image data
    v.IMAGES_DATA = {"Images":[]}
    func.write_json(v.IMAGES_DATA)
    # remove all images (both original and annoated)
    for image in images:
        uploaded_img = "static/uploads/"+image
        ann_img = "static/annotated_images/output_"+image
        # remove original image
        os.remove(uploaded_img)
        # remove anotated image
        os.remove(ann_img)
    return redirect(url_for("library"))

@app.route('/display/<filename>')
def display_image(filename, is_ann=False):
    """
    Displays Image (used in library and search page tables).
    
    :return: return the image to display
    """
    # check if the passed files are the original images or the annotated images
    is_ann = request.args.get("is_ann")
    # annotated image
    if is_ann == "True":
        return redirect(url_for('static', filename='annotated_images/' + filename))
    # original image
    else:
        return redirect(url_for('static', filename='uploads/' + filename))

@app.route("/search", methods=["POST", "GET"])
def search():
    """
    Search page. 
    
    :return: returns the search.html template
    """
    # set result to return the search.html template with necessary params if the user access the search page through url (no button clicks)
    result = render_template("search.html", trash_list=v.TRASH_LIST, selected_trash_list=v.SELECTED_TRASH_LIST, headings=v.SEARCH_TABLE_HEADERS, data=v.SEARCH_TABLE_ROWS, style="none", recyclable=v.RECYCLABLE_FILTER, non_recyclable=v.NON_RECYCLABLE_FILTER, quantity=v.QUANTITY_FILTER, quantityType=v.QUANTITY_TYPE_FILTER, intersection = v.INTERSECTION_FILTER)
    # if +, -, or search button was clicked
    if request.method == "POST":
        ############### + button ################
        # if + button clicked add selected trash category to the selected trash list
        if "+" in request.form:
            if "trash" in request.form:
                trash = request.form["trash"]
                v.TRASH_LIST.remove(trash)
                v.SELECTED_TRASH_LIST.append(trash)
        # if - button clicked remove selected trash category from the selected trash list and add it back to the trash categories
        if "-" in request.form:
            if "selectedtrash" in request.form:
                selectedtrash = request.form["selectedtrash"]
                v.SELECTED_TRASH_LIST.remove(selectedtrash)
                v.TRASH_LIST.append(selectedtrash)
        # update result to return the search.html template with necessary params if + or - button was clicked but not the search
        result = render_template("search.html", trash_list=v.TRASH_LIST, selected_trash_list=v.SELECTED_TRASH_LIST, headings=v.SEARCH_TABLE_HEADERS, data=v.SEARCH_TABLE_ROWS, style="none", recyclable=v.RECYCLABLE_FILTER, non_recyclable=v.NON_RECYCLABLE_FILTER, quantity=v.QUANTITY_FILTER, quantityType=v.QUANTITY_TYPE_FILTER, intersection = v.INTERSECTION_FILTER)
        ############# Search button ##############
        # if search button is clicked
        if "Search" in request.form:
            # reset predefined search table headers and rows
            v.SEARCH_TABLE_HEADERS = ["Images", "Quantity"]
            v.SEARCH_TABLE_ROWS = []
            # open json images data file
            with open(v.JSON_DATA_FILE,'r') as json_data:
                # load images data
                imgs_data = json.load(json_data)
                
                ############## Recyclables Filter ##############
                # if recyclables filter is checked
                if "Recyclables" in request.form:
                    r = request.form["Recyclables"]
                    v.SEARCH_TABLE_HEADERS.append(r)
                    v.RECYCLABLE_FILTER = "True"
                # if recyclables filter is not checked
                else:
                    v.NON_RECYCLABLE_FILTER = "False"
                    
                ############# Non-recyclables Filter #############
                # if non-recyclables filter is checked
                if "Non_recyclables" in request.form:
                    non_r = request.form["Non_recyclables"]
                    v.SEARCH_TABLE_HEADERS.append(non_r)
                    v.NON_RECYCLABLE_FILTER = "True"
                # if non-recyclables filter is not checked
                else:
                    v.NON_RECYCLABLE_FILTER = "False"

                ############# Trash Categories Filter #############
                # adds selected trash categories to the table headers
                for t in v.SELECTED_TRASH_LIST:
                    v.SEARCH_TABLE_HEADERS.append(t)
                # call checkClasses function to return a list of the names of
                # the images that has at least one of the selected trash categories
                classSet = func.checkClasses(v.SELECTED_TRASH_LIST, intersect)

                ################ Quantity Filter ################
                # if the user entered a quantity
                if "quantity" in request.form:
                    quantity = request.form["quantity"]
                # if the user entered a less than, greater than, or equal to associated with the quatity filter
                if "quantityType" in request.form:
                    quantityType = request.form["quantityType"]
                # initialize quantity set which will holds the names of the images that pass the quantity filter
                quantitySet = set()
                # if both quantity and quantity type have been entered by the user
                if quantity != "" and quantityType != "":
                    # call checkQuantity function to return a list of the names of the images that pass the quantity filter
                    quantitySet = func.checkQuantity(quantityType, quantity)
                
                ############## Intersection Filter ###############
                intersect = ""
                # check if intersection checkbox is checked
                if "Intersection" in request.form:
                    intersect = request.form["Intersection"]
                    v.INTERSECTION_FILTER = "True"
                # if intersection checkbox is not checked
                else:
                    v.INTERSECTION_FILTER = "False"

                ############## Merging Filters ###############
                # initialize finalSet which will contain the names of the images that fit all the entered filters by the user
                finalSet = set()
                # if a quantity or quantityType or trash categories have been entered or selected
                # and both quantitySet and classSet are empty then no images to display
                if len(quantitySet) == 0 and len(classSet) == 0 and (quantity != "" or quantityType != "" or len(v.SELECTED_TRASH_LIST) != 0):
                    finalSet = set()
                # else if both quantitySet and classSet are empty and no quantity or trash categories
                # have been entered or selected then display all images
                elif len(quantitySet) == 0 and len(classSet) == 0:
                    for image in imgs_data["Images"]:
                        finalSet.add(image["Name"])
                # if classSet is not empty and quantitySet is empty, but a quantity filter have been
                # entered then no images to display
                elif len(quantitySet) == 0 and len(classSet) != 0 and (quantity != "" or quantityType != ""):
                    finalSet = set()
                # if classSet is not empty and quantitySet is empty and no quantity filter have been
                # entered then display images in classSet
                elif len(quantitySet) == 0 and len(classSet) != 0:
                    finalSet = classSet
                # if quantitySet is not empty and classSet is empty, but one or more trash categories
                # have been selected then no images to display
                elif len(quantitySet) != 0 and len(classSet) == 0 and len(v.SELECTED_TRASH_LIST) != 0:
                    finalSet = set()
                # if quantitySet is not empty and classSet is empty and no trash categories have been
                # selected then display images in quantitySet
                elif len(quantitySet) != 0 and len(classSet) == 0:
                    finalSet = quantitySet
                # if non of the other conditions are met then dispaly the intersection of the quantitySet and classSet
                else:
                    finalSet = quantitySet.intersection(classSet)
                
                # for every image in the images data
                for image in imgs_data["Images"]:
                    img_data = []
                    # call get_classes_info which returns a dictionary with each image and its trash categories and corresponding count
                    classes_info = func.get_classes_info()
                    # for every image that's about to be displayed in the search table
                    for n in finalSet:
                        if image["Name"] == n:
                            # add the name and trash quantity of the image
                            img_data.append(image["Name"])
                            img_data.append(image["Quantity"])
                            
                            ################# recyblables count #################
                            r_count = 0
                            # check if recyblables filter is checked
                            if v.RECYCLABLE_FILTER == "True":
                                # counts the number of recyclable items in the imag
                                for c,c_count in classes_info[n].items():
                                    if c in v.RECYCLABLES:
                                        r_count+=c_count
                                img_data.append(r_count)
                            
                            ############### non-recyblables count ###############
                            nonr_count = 0
                            # check if non-recyblables filter is checked
                            if v.NON_RECYCLABLE_FILTER == "True":
                                # counts the number of non-recyclable items in the image
                                for c,c_count in classes_info[n].items():
                                    if c in v.NON_RECYCLABLES:
                                        nonr_count+=c_count
                                img_data.append(nonr_count)
                            
                            ########### selected trash categories count ###########
                            # counts the number of each selected trash category
                            for t in v.SELECTED_TRASH_LIST:
                                if t in classes_info[n]:
                                    img_data.append(classes_info[n][t])
                                else:
                                    img_data.append(0)
                    # add the rows to the table
                    v.SEARCH_TABLE_ROWS.append(img_data)
            # update result to return the search.html template with necessary params if search button was clicked
            result = render_template("search.html", trash_list=v.TRASH_LIST, selected_trash_list=v.SELECTED_TRASH_LIST, headings=v.SEARCH_TABLE_HEADERS, data=v.SEARCH_TABLE_ROWS, style="inline", recyclable=v.RECYCLABLE_FILTER, non_recyclable=v.NON_RECYCLABLE_FILTER, quantity=v.QUANTITY_FILTER, quantityType=v.QUANTITY_TYPE_FILTER, intersection = v.INTERSECTION_FILTER)
    return result

# Runs the web app
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.3', port=5000)
