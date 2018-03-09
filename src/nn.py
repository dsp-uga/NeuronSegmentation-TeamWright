import json
import matplotlib.pyplot as plt
from numpy import array, zeros
from scipy.misc import imread
from scipy.misc import imsave
import scipy
from PIL import Image
from glob import glob
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(7)
np.set_printoptions(threshold=np.nan)

#####################################################################
#Data 'X' Processing
files = sorted(glob('images1/*.tiff'))
imgs = array([imread(f) for f in files]) #Raw image data

a=0
b=1
rgbNormalmax=0
rgbNormalmin=255
imgs= a+(((imgs-rgbNormalmin)*(b-a))/(rgbNormalmax-rgbNormalmin))

t_data = imgs[0:int(imgs.shape[0]*0.50)] #70 percent data
print "Train shape",t_data.shape
v_data =imgs[int(imgs.shape[0]*0.50)+1: ] #30 percent data
print "Validation shape",v_data.shape
def prepare(imgs):
    t_imgs = np.transpose(imgs)
    tod_data = t_imgs.reshape(imgs.shape[1]*imgs.shape[2], imgs.shape[0])
    return tod_data

train_data = prepare(t_data)
val_data = prepare(v_data)

#####################################################################
#Labels 'Y'  Processing
with open('regions/regions.json') as f:
    regions = json.load(f)

dims = imgs.shape[1:]

mask = zeros(dims)

coordinates = []
for s in regions:
    coordinates.extend(list(s['coordinates']))

to_label =  list(coordinates)

mask[zip(*to_label)] = 1

labels = mask.reshape(mask.shape[0]*mask.shape[1], 1)

print "Labels ",labels.shape

##################################################################
#print "Train data 0",train_data[0]

###################################################################
#Model
'''
model = Sequential()
model.add(Dense(2000, input_dim=train_data.shape[1], activation='relu'))
model.add(Dense(1800, activation='relu'))
model.add(Dense(1600, activation='relu'))
model.add(Dense(1400, activation='relu'))
model.add(Dense(1200, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(600, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
'''


#################################################################
#Sample model

model = Sequential()
model.add(Dense(1000, input_dim=train_data.shape[1], activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#####################################################################
#Compiling model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#####################################################################
#Fitting
model.fit(train_data, labels, epochs=150, batch_size=50,verbose=1)

#####################################################################
# Evaluate the model
scores = model.evaluate(val_data, labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#####################################################################
# Calculate predictions
predictions = model.predict(train_data)

#####################################################################
# Round predictions
bwImage = [int(round(x[0])) for x in predictions]
bwImage = bwImage.reshape(512,512)

####################################################################
#Extracting neurons to list
def blobs(img):
    scipy.misc.imsave('outfile.jpg', img)

    im = imread("outfile.jpg",cv2.IMREAD_GRAYSCALE)

    im2, contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Detect blobs.
print "Blobs list ",blobs(bwImage)
