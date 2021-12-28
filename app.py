# Eviroment Libraries for our use
from flask import Flask, request, jsonify, url_for, render_template
import uuid
import os
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
import cv2


# Extension of images that are possible for  prediction
ALLOWED_EXTENSION  =set(['txt', 'pdf', 'png','jpg','jpeg','gif'])

# parameters for preprocessing
IMAGE_HEIGHT =256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3
# Prediction Labels
label_names = {0 : 'Covid-19', 1 : 'Lung_Opacity' , 2: 'Normal', 3 : 'Viral Pneumonia'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSION

# define the flask app and load pretrained model
app = Flask(__name__)
model = load_model("Rnet_Model.h5")
# HTML templete
@app.route('/')
def index():
    return render_template('ImageML.html')
#
@app.route('/api/image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return render_template('ImageML.html', prediction='No posted image. Should be attribute named image')
    file = request.files['image']
    
    if file.filename =='':
        return render_template('ImageML.html', prediction = 'You did not select the image')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print("***"+filename)
        x = []
        # loading image file
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        img = Image.open(BytesIO(file.read()))
        img.load()
        #preprocessing the image file
        img = image.img_to_array(img)
        if img.shape[2] == 3:
            img = cv2.resize(img,(2,256))
            img = img / 255

            img = img.reshape(-1,256,256,3)
            predict = model.predict(img)
            predict = np.argmax(predict)
        
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img,(256,256))
            
            img = img / 255
            
            img = img.reshape(-1,256,256,3)
            predict = model.predict(img)
            predict = np.argmax(predict)
        
        
        response = (label_names[predict])
        return render_template('ImageML.html', prediction = '{}'.format(response))
    else:
        return render_template('ImageML.html', prediction = 'Invalid File extension')
# return the prediction to above templete
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
