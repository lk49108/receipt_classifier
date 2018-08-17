import constants as CONST
from models import SGANModelWithSpatialTransformer
import numpy as np
import argparse
import os
import datasets
import cv2

def importData():
	#construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--folder", required = True,
	help = "Path to the root folder has to be provided")
	rootDir = os.path.normpath(vars(ap.parse_args())['folder']) 

	#do not allow access to 'mozgalo' folder
	#if rootDir == 'mozgalo':
	#	print('Not allowed to change provided directory')
	#	exit()
	
	trainingData = datasets.loadTrainingSet(rootDir, CONST.classes, validation_size = CONST.validation_size)
	
	print("Number of files in Training-set:\t{}".format(len(trainingData.train.labels)))
	print("Number of files in Validation-set:\t{}".format(len(trainingData.valid.labels)))
	
	return trainingData


def main():
	trainingData = importData()
	
	model = SGANModelWithSpatialTransformer()
	model.train(trainingData = trainingData, num_iteration = 5000000)
	
if __name__ == '__main__':
	main()