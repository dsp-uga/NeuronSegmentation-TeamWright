import json
import matplotlib.pyplot as plt
from numpy import array, zeros
from scipy.misc import imread
from glob import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

np.set_printoptions(threshold=np.nan)
files = sorted(glob('images1/*.tiff'))
imgs = array([imread(f) for f in files])

t_imgs = np.transpose(imgs)

tod_data = t_imgs.reshape(imgs.shape[1]*imgs.shape[2], imgs.shape[0])

print "Two dimensional data shape ",tod_data.shape



'''
print "Original images shape ",imgs.shape
print "Transposed images shape", t_imgs.shape

print "Images"
print imgs
print "Transpose"
print t_imgs
'''

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

#Model
model = Sequential()
model.add(Dense(2000, input_dim=imgs.shape[0], activation='relu'))
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

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(tod_data, labels, epochs=50, batch_size=50)

# evaluate the model
scores = model.evaluate(tod_data, labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# calculate predictions
predictions = model.predict(tod_data)
# round predictions
rounded = [round(x[0]) for x in predictions]

print rounded.index(1.0)
