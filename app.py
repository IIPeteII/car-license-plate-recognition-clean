#This app ties together all our work

#------------ Import libraries

#COMMON
import pandas as pd #DF work
import numpy as np #Functions
import matplotlib.pyplot as plt #Visualizations
import requests
import altair as alt #Visualizations
import io #Buffers for DF's
from io import BytesIO #BytesIO
from io import StringIO #StringIO
import http.client #API
import os #operating system functions
from PIL import Image #open pictures
from pathlib import Path #path function
from scipy.io import loadmat #load .mat files
import datetime #dates and time stuff
import json

#uploading/file management
from tempfile import NamedTemporaryFile

#ML/Computer Vision
import cv2 #computer vision
import tensorflow as tf #tensorflow
from keras.preprocessing.image import ImageDataGenerator #generate
from keras.models import Sequential #sequential model
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D #model functions
from keras import optimizers #model optimizers such as Adam and learning rates

#Deployment
import streamlit as st #app deployment

#Introduction

st.title('DSBA - Exam project :car:')

#image of a cool car
image_url = requests.get('https://raw.githubusercontent.com/IIPeteII/car-license-plate-recognition-clean/main/app-images/app_lambo.jpg')
app_image = Image.open(BytesIO(image_url.content))
st.image(app_image, caption='Credits to https://unsplash.com/@reinhartjulian for the picture')

#header for the project
st.header('What does it do?')
st.markdown('This is an app that _regonizes_ and **detects** license plates in images, then return an image of the car model.\
    The app takes the string of the license plate and searches for it in the Danish license plate database, which then returns information such as brand, model and year.\
    Thereafter, the app goes through a large dataset of car models and returns an image of the car, if the car is not in the dataset - a google search is performed.')

#------------ Data input
st.subheader('Input a picture')
st.markdown('Upload your picture in the box below, or take a picture with your phone')

#upload a picture
uploaded_file = st.file_uploader("Upload your picture (only .jpg)", type=["jpg"])
if uploaded_file is not None:
    # Display image
    st.write("Original Image")
    st.image(uploaded_file, caption="Uploaded Image")

    # Convert image to cv2 format
    bytes_data = uploaded_file.getvalue()
    opencv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # opencv_image is now an array that can be processed with OpenCV
    st.write("`type(opencv_image)`", type(opencv_image))
    st.write("`opencv_image.shape`", opencv_image.shape)

#------------ License plate detection model
st.subheader('License plate detection model')

#create extraction function

def extract_plate(img): # the function detects and perfors blurring on the number plate.
	plate_img = img.copy()
	
	#Loads the data required for detecting the license plates from cascade classifier.
	plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')

	# detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
	plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7)

	for (x,y,w,h) in plate_rect:
		plate = plate_img[y:y+h, x:x+w, :]
		# finally representing the detected contours by drawing rectangles around the edges.
		cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,51,255), 3)
        
	return plate_img, plate # returning the processed image.

#Apply extraction function

dk_test_img = cv2.imread(uploaded_file) #read file
plate_img_out, plate_out = extract_plate(dk_test_img) #apply

#Match contours



#Find characters function



#create fit parameters



#model



#logs



#model training

#CAREFUL THIS IS A LARGE PROCESS



#print plate number



#show prediction



#------------ API-integration to database
st.subheader('API-call from Danish license plate database')



#------------ Picture database output
st.subheader('Picture of car from dataset')

#------------ Google search
st.subheader('Link to google of car')
#st.button('press here to see the car')