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
uploaded_file = st.file_uploader("Upload a file", type="jpg")
if uploaded_file is not None:
    # To read file as bytes:
    st.image(uploaded_file, caption='Your uploaded picture')
    #bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    #string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    #dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)

#take a picture
#img_file_buffer = st.camera_input("Take a picture")

#if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    #bytes_data = img_file_buffer.getvalue()
    #cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Check the type of cv2_img:
    # Should output: <class 'numpy.ndarray'>
    #st.write(type(cv2_img))

    # Check the shape of cv2_img:
    # Should output shape: (height, width, channels)
    #st.write(cv2_img.shape)

#------------ License plate detection model
st.subheader('License plate detection model')

#Extract plate function
def extract_plate(img): # the function detects and perfors blurring on the number plate.
	plate_img = img.copy()
	
	#Loads the data required for detecting the license plates from cascade classifier.
	plate_cascade = cv2.CascadeClassifier('indian_license_plate.xml')

	# detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
	plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.3, minNeighbors = 7)

	for (x,y,w,h) in plate_rect:
		#a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1])) #parameter tuning #Check this later!
		#plate = plate_img[y+a:y+h-a, x+b:x+w-b, :]
		plate = plate_img[y:y+h, x:x+w, :] #don't need the parameter tuning stuff
		# finally representing the detected contours by drawing rectangles around the edges.
		cv2.rectangle(plate_img, (x,y), (x+w, y+h), (51,51,255), 3)
        
	return plate_img, plate # returning the processed image.

#Apply extraction function
dk_test_img = cv2.imread(uploaded_file) #set to work with uploaded file currently
plate_img_out, plate_out = extract_plate(dk_test_img)

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

input_plate = print(show_results) #Create variable from model
link = "/vehicles?registration_number={}".format(input_plate)
print (link)

#------------ Picture database output
st.subheader('Picture of car from dataset')

#------------ Google search
st.subheader('Link to google of car')
#st.button('press here to see the car')