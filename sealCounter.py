#!/usr/bin/python2.7

from skimage import io, filters
from skimage.feature import hog
from skimage.color import rgb2gray

from scipy.misc import imread
import scipy.misc
from scipy import ndimage
from scipy import misc

from sklearn.ensemble import RandomForestClassifier

import os,sys
from PIL import Image
import numpy as np
import time

def preprocessImages():
	start_time = time.time()
	#imageFile = open('/home/cpsc/MachineLearning/ProcessedKaggle/', 'w+')
	#print('yes', imageFile)
	imgDest = '/home/cpsc/MachineLearning/ProcessedKaggle/'
	for filename in os.listdir('/home/cpsc/MachineLearning/Train'):
		img = imread('/home/cpsc/MachineLearning/Train/'+filename)
		img = rgb2gray(img)
		#print(np.shape(img))	
		#edges = filters.sobel(img)
		sobel = filters.sobel_h(img)
		#hog_image = hog(img)#, orientations = 8, pixels_per_cell=(16,16), cells_per_block(1,1), visualise=True)
		#scipy.misc.imsave(imgDest + 'edges'+ filename , edges)
		scipy.misc.imsave(imgDest + 'sobel'+ filename, sobel)
	print("--- %s seconds --- to preprocess" % (time.time()- start_time))

def sealForrest():
	data = np.array([],dtype=float)
	target = np.array([],dtype=float)
	start_time = time.time()
	for filename in os.listdir('/home/cpsc/MachineLearning/ProcessedKaggle'):
		if('sobel' in filename):
			img = imread('/home/cpsc/MachineLearning/ProcessedKaggle/'+filename)
			img = img_to_matrix(img)
			img = flatten(img)
			data = np.append(data,img)
	forrest = RandomForestClassifier(n_estimators = 10)
	csvData = np.loadtxt('train.csv', delimiter = ',',skiprows=1).astype(np.float)
	csvData = np.array(csvData)
	#print(target[41:50,1:])
	#print(np.sum(target[41:50,1:]))
	for i in range(40,50):
		target = np.append(target,np.sum(csvData[i:i+1,1:]))
		
	print(data)
	print(np.ravel(np.asarray(target)))
	target = np.array(target,dtype=np.float)
	print(np.shape(data), np.shape(target),data.dtype,target.dtype)
	forrest.fit(data,target)
	#forrest.fit(np.ravel(np.asarray(data)),np.ravel(np.asarray(target,dtype=np.float32)))	
		
	print("--- %s seconds --- for Random Forrest" % (time.time()-start_time))

def img_to_matrix(img, verbose=False):
	#img = Image.open(filename)
	if verbose == True:
		print "changing size from %s to %s" % (str(img.size),"standard size")
	#img = list(img.getdata())
	#img = map(list,img)
	img = np.array(img)
	return img

def flatten(img):
	s = img.shape[0] * img.shape[1]
	img_wide = img.reshape(1,s)
	return img_wide[0]

#preprocessImages()
sealForrest()
