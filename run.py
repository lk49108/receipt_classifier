import constants as CONST
from models2 import SGANModelWithSpatialTransformer
import numpy as np
import argparse
import os
import datasets
import cv2

def main():
	model = SGANModelWithSpatialTransformer()
	model.run()
	
	
if __name__ == '__main__':
	main()