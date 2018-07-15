from flask import Flask,request
import sys
import os
import base64

#import detect

from io import BytesIO
from PIL import Image
#import numpy as np

#Crucial for saving and handling post form data and files
from werkzeug.datastructures import ImmutableMultiDict

from flask import jsonify
import json

app = Flask(__name__)



@app.route("/", methods = ['POST'])
def index():
    print('Request-form',list(request.form.keys()),file=sys.stderr)
    print('Request-form-name',request.form['name'],file=sys.stderr)
    # print('Request-form-image',request.form['image'],file=sys.stderr)
    
    image_name = request.form['name']
    image_string = request.form['image']
    #image_bytes = bytes(image_string,'utf-8')
    #image_decoded = base64.decodestring(image_string)
    
    image = Image.open(BytesIO(base64.b64decode(image_string)))
        
    result_image = image.rotate(270,expand=True)

    
#     input_array = np.array(rotated_image) 
    
#     input_array = np.expand_dims(input_array,axis=0)
    
#     result_array = detect.run(input_array)
    
#     result_image = Image.fromarray(result_array)
    
    #print('rotated_image.shape = ',input_array.shape)
    
    #rotated_image.save('default.jpg',format='JPEG')
    
    #convert image back to string..
    buffered = BytesIO()
    result_image.save(buffered, format="JPEG")
    final_img_str = base64.b64encode(buffered.getvalue())

    
    #     print('Request-files:',request.files,file=sys.stderr)
#     print('Requestfiletype:',type(request.files),file=sys.stderr)

#     data = request.files.to_dict()
   
#     print('data',data,file=sys.stderr)
   
#     #to-do Input file validation... (ensure input file is valid jpg or png)
#     file = data['upload']
    
#     print('File name:',file.filename,file=sys.stderr)
    
#     file_path = os.path.join("Images",file.filename)
  
#     file.save(file_path)
    
#     print('File saved with name:',file.filename,file=sys.stderr)
    
    #Deserialize the image..
#     with open(image_name,'wb') as image_file:
#         image_file.write(image)
    
    response = final_img_str
    
    return response

if(__name__ == "__main__"):
    app.run(host = '0.0.0.0',port = 5000)