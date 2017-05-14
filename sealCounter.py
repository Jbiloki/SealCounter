#!/usr/bin/python2.7

from skimage import io, filters
from skimage.feature import hog
from skimage.color import rgb2gray

from scipy.misc import imread
import scipy.misc
from scipy import ndimage
from scipy import misc

from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import glob
import cv2

import matplotlib.pyplot as plt
#%matplotlib inline

import os,sys
from PIL import Image
import numpy as np
import time

def preprocessImages():
	start_time = time.time()
	#imageFile = open('/home/cpsc/MachineLearning/ProcessedKaggle/', 'w+')
	#print('yes', imageFile)
	imgDest = '/home/cpsc/MachineLearning/ComputerVision/KaggleCode/ProcessedKaggle/gauss'
	for filename in os.listdir('/home/cpsc/MachineLearning/ComputerVision/KaggleCode/Train/'):
		img = imread('Train/'+filename)
		img = rgb2gray(img)
		#print(np.shape(img))	
		#edges = filters.sobel(img)
		gaus = filters.gaussian(img)
		#sobel = filters.sobel_h(gaus)
		#hog_image = hog(img)#, orientations = 8, pixels_per_cell=(16,16), cells_per_block(1,1), visualise=True)
		#scipy.misc.imsave(imgDest + 'edges'+ filename , edges)
		scipy.misc.imsave(imgDest + 'gause'+ filename, gaus)
	print("--- %s seconds --- to preprocess" % (time.time()- start_time))

def sealForrest():
	data = np.array([],dtype=float)
	target = np.array([],dtype=float)
	start_time = time.time()
	for filename in os.listdir('/home/cpsc/MachineLearning/ComputerVision/KaggleCode/ProcessedKaggle'):
		if('sobel' in filename):
			img = imread('/home/cpsc/MachineLearning/ComputerVision/KaggleCode/ProcessedKaggle/'+filename)
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


def sealCV():
	train_data = pd.read_csv('train.csv')
	train_imgs = sorted(glob.glob('ProcessedKaggle/gaussgause*.jpg'))# key = lambda name: int(os.path.basename(name)[:-4]))
	train_dot = sorted(glob.glob('Kaggle/TrainSmall2/TrainDotted/*.jpg'))# key = lambda name: int(os.path.baseename(name)[:-4]))
	#submission = pd.read_csv('~/MachineLearning/KaggleCode/sample_submission.csv')
	print(train_data.shape)
	print('Number of Train Images: {:d}'.format(len(train_imgs)))
	print('Number of Dotted-Train Images: {:d}'.format(len(train_dot)))
	print(train_data.head(6))
	idx = 2
	image = cv2.cvtColor(cv2.imread(train_imgs[idx]),cv2.COLOR_BGR2RGB)
	image_dot = cv2.cvtColor(cv2.imread(train_dot[idx]),cv2.COLOR_BGR2RGB)
	img = image[1350:1900, 3000:3400]
	img_dot = image_dot[1350:1900, 3000:3400]
	diff = cv2.absdiff(image_dot,image) #get the difference of the cropped images
	gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
	ret, th1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU) #threash hold at 0 to further differentiate the ground from the seals
	cnts = cv2.findContours(th1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	#print(cnts)
	print("Sea Lions Found: {}".format(len(cnts)))
	
#preprocessImages()
#sealForrest()
sealCV()
