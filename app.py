#This app ties together all our work

#------------ Import libraries

#COMMON
import pandas as pd #DF work
import numpy as np #Functions
#import matplotlib.pyplot as plt #Visualizations
import altair as alt #Visualizations
import io #Buffers for DF's
import http.client #API
import os #operating system functions
#from selenium import webdriver #google search
#import chromedriver_autoinstaller #chrome driver to open a browser
#from PIL import image #open pictures
#from pathlib import Path #path function
#from scipy.io import loadmat #load .mat files
#import datetime #dates and time stuff

#ML/Computer Vision
#import cv2 #computer vision
#import tensorflow as tf #tensorflow
#from keras.preprocessing.image import ImageDataGenerator #generate
#from keras.models import Sequential #sequential model
#from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D #model functions
#from keras import optimizers #model optimizers such as Adam and learning rates

#Deployment
import streamlit as st #app deployment

#Introduction

st.title('DSBA 2022 - Car number plate detection')
st.header('detecting license plates and returning an image of the car')


#------------ License plate detection
st.subheader('license plate detection model')

#Need a contigency as to whether we upload or take a picture!
#st.text, st.camera_input("Take a picture"), st.file_uploader("Upload picture")


#------------ API-integration to database
st.subheader('API-call from Danish license plate database')

#------------ Picture database output
st.subheader('Picture of car from dataset')

#------------ Google search
st.subheader('Link to google of car')
#st.button('press here to see the car')

st.write()