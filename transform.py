# import the necessary packages

from skimage.filters import threshold_local
import cv2
import argparse
import os
import shutil	
import imutils

def convert(file):

	# load the image and compute the ratio of the old height
	# to the new height, clone it, and resize it
	image = cv2.imread(file)
	image = cv2.resize(image, (256, 512))

	#Turns image into black-white image
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	T = threshold_local(image, 11, offset = 10, method = "gaussian")
	image = (image > T).astype("uint8") * 255
	
	#Creating transformed image
	imageDest = str(file).replace('.jpg', '_black_white.jpg')
	cv2.imwrite(imageDest, image)
	
	#Removing the old one
	os.remove(file)
	
if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--folder", required = True,
	help = "Path to the root folder has to be provided")
	rootDir = os.path.normpath(vars(ap.parse_args())['folder']) 

	#if rootDir == 'mozgalo':
	#	print('Not allowed to change provided directory')
	#	exit()
	
	#if rootDir == './'
	
	#COPIES file dir##############################################	
	#rootDirTransformed = os.path.normpath(str(rootDir) + 'a')
	#if not os.path.exists(rootDirTransformed):
	#	os.makedirs(rootDirTransformed)
	#
	#
	#def copyTree(src, dst, symlinks=False, ignore=None):
	#   for item in os.listdir(src):
	#        s = os.path.join(src, item)
	#       d = os.path.join(dst, item)
	#        if os.path.isdir(s):
	#            shutil.copytree(s, d, symlinks, ignore)
	#        else:
	#            shutil.copy2(s, d)
	##############################################################
	#copyTree(rootDir, rootDirTransformed)	
	##############################################################

for subdir, dirs, files in os.walk(rootDir):
	for file in files:
		convert(os.path.join(subdir, file))	
	