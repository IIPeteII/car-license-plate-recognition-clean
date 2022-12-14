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
import pickle
import cv2 #computer vision
import tensorflow as tf #tensorflow
from tensorflow import keras
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

#create extraction function honestly don't know how much of this is needed

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

#dk_test_img = cv2.imread(opencv_image) #read file
test_bytes_data = uploaded_file.getvalue()
dk_test_img = cv2.imdecode(np.frombuffer(test_bytes_data, np.uint8), cv2.IMREAD_COLOR) #read file
plate_img_out, plate_out = extract_plate(dk_test_img) #apply

#Match contours
# Match contours to license plate or character template
def find_contours(dimensions, img) :

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv2.imread('contour.jpg')
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        #detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        #checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            #extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')

#           Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) #List that stores the character's binary image (unsorted)
            
    #Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    plt.show()
    #arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res


#Find characters function
# Find characters in the resulting images
def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/10,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

char = segment_characters(plate_out) #showing plates
#create fit parameters

#train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
#train_generator = train_datagen.flow_from_directory(
        #r'data/data/train',  # this is the target directory
        #target_size=(28,28),  # all images will be resized to 28x28
        #batch_size=1,
        #class_mode='categorical')

#validation_generator = train_datagen.flow_from_directory(
        #r'data/data/val',  # this is the target directory
        #target_size=(28,28),  # all images will be resized to 28x28        
        #batch_size=1,
        #class_mode='categorical')


#model


model = keras.models.load_model(r'Model')

#model = Sequential()
#model.add(Conv2D(32, (24,24), input_shape=(28, 28, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.4))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dense(36, activation='softmax'))

#model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.00001), metrics=['accuracy'])
#logs

#class stop_training_callback(tf.keras.callbacks.Callback):
  #def on_epoch_end(self, epoch, logs={}):
    #if(logs.get('val_acc') > 0.995):
      #self.model.stop_training = True

#log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#model training

#CAREFUL THIS IS A LARGE PROCESS
#batch_size = 1
#callbacks = [tensorboard_callback, stop_training_callback()]
#model_history = model.fit(
      #train_generator,
      #steps_per_epoch = train_generator.samples // batch_size,
      #validation_data = validation_generator, 
      #validation_steps = validation_generator.samples // batch_size,
      #epochs = 80)

#print plate number

def fix_dimension(img): 
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
  
def show_results():
    dic = {}
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(char): #iterating over the characters
        img_ = cv2.resize(ch, (28,28))
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        y_ = np.argmax(model.predict(img)[0]) #predicting the class np.argmax(model.predict(x_test), axis=-1)     classes_ = np.argmax(y_, axis = 1)
        character = dic[y_] #
        output.append(character) #storing the result in a list
        
    plate_number = ''.join(output)
    
    return plate_number

final_plate = show_results()
print(final_plate)

#show prediction
fig = plt.figure(figsize=(10,6))
for i,ch in enumerate(char):
    img = cv2.resize(ch, (28,28))
    plt.subplot(3,4,i+1)
    plt.imshow(img,cmap='gray')
    plt.title(f'predicted: {show_results()[i]}')
    plt.axis('off')
plt.show()
st.pyplot(fig)

#------------ API-integration to database
st.subheader('API-call from Danish license plate database')

#api-script
final_plate #Create variable from model
link = "/vehicles?registration_number={}".format(final_plate)
print (link)

conn = http.client.HTTPSConnection("v1.motorapi.dk")
payload = ''
headers = {
  'X-AUTH-TOKEN': 'gq59xnw6jombh3vpiuvv0lzfh8h7df36'
}
conn.request("GET", link , payload, headers) #We add the link to automate the process of connecting the car plate number with the API
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))

#transform to JSON

my_json = data.decode('utf8').replace("'", '"')
print(my_json)
print('- ' * 20)

# Load the JSON to a Python list & dump it back out as formatted JSON
data = json.loads(my_json)
s = json.dumps(data, indent=4, sort_keys=True)

#create dictionary

d1=dict(enumerate(data))

#return car type
car_type = (f"{(data[0]['make'])} {(data[0]['model'])} {(data[0]['first_registration'])}")

#return car year
car_year = data[0]['first_registration']
car_year2 = car_year[:4]
car_year = car_year2

#final car_type

car_type = (f"{(data[0]['make'])} {(data[0]['model'])} {car_year}")
car_type = car_type.lower()
print(car_type)

#------------ Google search
st.subheader('Link to google of car')
#st.button('press here to see the car')

def google_search(url):
  url = url.replace(' ', '+')
  link = "https://www.google.com/search?tbm=isch&q=" + url
  return link

st.write(google_search(car_type))