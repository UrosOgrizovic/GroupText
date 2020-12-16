import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, json
from werkzeug.utils import secure_filename
from werkzeug.serving import run_simple
from detect_document_text_in_image import detect_document, render_doc_text
from PIL import Image
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'static/'
OUTPUT_FILE_PATH = UPLOAD_FOLDER + 'uploaded_img_res.jpg'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
IMAGE_MAX_WIDTH = 800

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def resize_image(img):
    if img.size[0] <= IMAGE_MAX_WIDTH:  # no need to enlarge
        return img
    wpercent = (IMAGE_MAX_WIDTH / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    return img.resize((IMAGE_MAX_WIDTH, hsize), Image.ANTIALIAS)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_image(path):
    with open(path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return render_template("index.html")
        if file and allowed_file(file.filename):
            # filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_img.jpg')
            file.save(path)
            render_doc_text(path, OUTPUT_FILE_PATH)
            result = detect_document(path)
            resize_image(Image.open(OUTPUT_FILE_PATH)).save(OUTPUT_FILE_PATH)
            image_string = read_image(OUTPUT_FILE_PATH)
            return app.response_class(response=json.dumps({'result': result, 'image': image_string}),
                                      status=200,
                                      mimetype='application/json')
    return render_template("index.html")
    # return '''
    # <!doctype html>
    # <title>Upload new File</title>
    # <h1>Upload new File</h1>
    # <form method=post enctype=multipart/form-data>
    #   <input type=file name=file>
    #   <input type=submit value=Upload>
    # </form>
    # '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


# @app.route("/")
# def hello():
#     return "Hello World!"


if __name__ == "__main__":
    # app.run()  # Flask
    run_simple("localhost", 5000, app)  # Werkzeug
