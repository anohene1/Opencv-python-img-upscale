from cv2 import dnn_superres
import cv2
import os
from flask import Flask
from flask import send_file
import urllib.request
from datetime import datetime

myapp = Flask(__name__)


@myapp.route('/')
def imageProcess():
    imgName = str(datetime.now())+".jpg"
    imgURL = 'https://media.geeksforgeeks.org/wp-content/uploads/20190802022327/Annotation-2019-08-02-022111.png'
    urllib.request.urlretrieve(imgURL, imgName)

    # Create an SR object - only function that differs from c++ code
    sr = dnn_superres.DnnSuperResImpl_create()
    # Read image
    image = cv2.imread(imgName)
    # Read the desired model
    path = "FSRCNN_x2.pb"
    sr.readModel(path)
    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("fsrcnn", 2)
    # Upscale the image
    result = sr.upsample(image)
    # Save the image
    cv2.imwrite(imgName, result)

    # save the image to memory
    ImgToSend = send_file(imgName, attachment_filename=imgName)

    # remove the local image
    os.remove(imgName)

    return ImgToSend
