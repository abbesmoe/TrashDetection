from flask import Flask, redirect, url_for, render_template, request, flash, send_file
import urllib.request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

trash_list = ["Plastic","Cardboard","Aluminium"]
selected_trash_list = []
uploads_list = []

headings = ["Images","Quantity","Recyclables"]
data = [
    ["img100.jpg","5","Plastic Bottles"],
    ["img101.jpg","7","Aluminium foil, Paper"],
    ["img102.jpg","11","Cardboard"],
    ["img102.jpg","11","Cardboard"]
]

UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['jpg','png','jpeg','img','tif','tiff','bmp','gif','eps','raw','mp4','mov','wmv','flv','avi'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/download', methods=['GET'])
def download_file():
    if request.method=="GET":
        p = "static/uploads/"+request.args.get("img")
        return send_file(p,as_attachment=True)
    path = "static/uploads"
    images = os.listdir(path)
    return render_template("library.html", images=images)

@app.route('/remove', methods=['GET'])
def remove_file():
    if request.method=="GET":
        filename = "static/uploads/"+request.args.get("img")
        os.remove(filename)
    path = "static/uploads"
    images = os.listdir(path)
    return render_template("library.html", images=images)

@app.route("/library")
def library():
    path = "static/uploads"
    images = os.listdir(path)
    return render_template("library.html", images=images)    

@app.route("/uploadredirect")
def uploadredirect():
    return redirect(url_for("upload"))

@app.route("/upload")
def upload():
    return render_template("upload.html")

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
    return render_template('upload.html', filename=filename)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route("/searchredirect")
def searchredirect():
    return redirect(url_for("search"))

@app.route("/search", methods=["POST", "GET"])
def search():
    global selected_trash_list
    global trash_list
    global headings
    result = render_template("search.html", trash_list=trash_list, selected_trash_list=selected_trash_list, headings=headings, data=data, style="none")
    if request.method == "POST":
        if "+" in request.form:
            if "trash" in request.form:
                trash = request.form["trash"]
                trash_list.remove(trash)
                selected_trash_list.append(trash)
                result = render_template("search.html", trash_list=trash_list, selected_trash_list=selected_trash_list, headings=headings, data=data, style="none")
        elif "-" in request.form:
            if "selectedtrash" in request.form:
                selectedtrash = request.form["selectedtrash"]
                selected_trash_list.remove(selectedtrash)
                trash_list.append(selectedtrash)
                result = render_template("search.html", trash_list=trash_list, selected_trash_list=selected_trash_list, headings=headings, data=data, style="none")
        elif "Search" in request.form:
            headings = ["Images","Quantity","Recyclables"]
            quantity = ""
            quantityType = ""
            if "type" in request.form:
                type = request.form["type"]
                if type != "":
                    headings.append(type)
            for t in selected_trash_list:
                headings.append(t)
            if "quantity" in request.form:
                quantity = request.form["quantity"]
            if "quantityType" in request.form:
                quantityType = request.form["quantityType"]
            if quantity != "" and quantityType != "":
                #handle quantity
                print("")
            result = render_template("search.html", trash_list=trash_list, selected_trash_list=selected_trash_list, headings=headings, data=data, style="inline")
    return result

if __name__ == "__main__":
    app.run(debug=True)
