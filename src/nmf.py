"""
would need to install following packages
- numpy
- scipy
- scikit-learn
"""

import os
import numpy as np
from scipy import misc
# import nimfa
from sklearn.decomposition import NMF

import time


path = "/home/vinay/UGA/Acad/Spring18/DSP/project3/data/training/neurofinder.00.00/images/"
V = np.array([])

fcount = 0

stime = time.time()
# reading images from file
for fname in os.listdir(path):
	new_image = misc.imread(path + fname, flatten='true', mode = 'L')
	#print(np.shape(new_image))
	# reshaping them to (1, mxn)
	img = np.reshape(new_image, new_image.shape[0]*new_image.shape[1])
	if fcount%100 == 0:
		print(fcount, " images read")
	# making a matrix of (k, mxn) for k images in dataset
	if fcount > 0:
		V = np.vstack((V, img))
	else:
		V = img
#	print(img)
#	if fcount == 800:
#		break
	fcount += 1

etime = time.time()
print("time taken to read images: ", etime - stime)
print(np.shape(V))

stime = time.time()
# need to take a transpose so that each column represents an image and no. of columns represent no. of images
V = np.transpose(V)
etime = time.time()
print("time taken for transpose: ", etime - stime)
print(np.shape(V))

"""
stime = time.time()
# have no idea. just use this reference: http://nimfa.biolab.si/nimfa.methods.factorization.lfnmf.html
# lfnmf = nimfa.Lfnmf(V, rank = np.shape(V)[1]/2, max_iter = 15, test_cov = 3)
lfnmf = nimfa.Lfnmf(V, max_iter = 15, test_cov = 3)
lfnmf_fit = lfnmf()
etime = time.time()
print("time taken for lfnmf for ", np.shape(V), " matrix is: ", etime - stime)
W = lfnmf.basis()
H = lfnmf.coef()
print(W)
print("W.dimensions: ", np.shape(W))
print("H.dimensions: ", np.shape(H))
"""

# no idea again. reference: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
model = NMF(n_components = 50, random_state = 0)
print("NMF started")
stime = time.time()
W = model.fit_transform(V)
etime = time.time()
print("time for fit_transform: ", etime - stime)
stime = time.time()
H = model.n_components_
etime = time.time()
print("time for components: ", etime - stime)
