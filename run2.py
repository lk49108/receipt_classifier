import tensorflow as tf
import cv2
import numpy as np
import os
import csv
from scipy import ndimage

MODEL = '/leonardo_model-265000.meta'
MODELS = './models'
ROOT_PATH = './test_set'

classes = [
	'Albertsons',
	'BJs',
	'Costco',
	'CVSPharmacy',
	'FredMeyer',
	'Frys',
	'HarrisTeeter',
	'HEB',
	'HyVee',
	'JewelOsco',
	'KingSoopers',
	'Kroger',
	'Meijer',
	'Publix',
	'Safeway',
	'SamsClub',
	'ShopRite',
	'Smiths',
	'StopShop',
	'Target',
	'Walgreens',
	'Walmart',
	'Wegmans',
	'WholeFoodsMarket',
	'WinCoFoods',
	'Other'
	]

with tf.Session() as sess:
	#Loading trained network
	new_saver = tf.train.import_meta_graph(MODELS + MODEL)
	new_saver.restore(sess, tf.train.latest_checkpoint(MODELS))
	
	#Accessing the graph
	graph = tf.get_default_graph()
	
	#Accessing x (input) tensor and y_pred (prediction) tensor
	x = graph.get_tensor_by_name("x:0")
	y_pred = graph.get_tensor_by_name("discriminator/d_y_pred:0")
	
	#for cls, predVal in zip(classes, predictionArray[0]):
	#	print(cls, ' ', predVal)
	#print(predictionArray[0])
	#print()
	
	#for cls, predVal in zip(classes, predictionArray[0]):
	#	if predVal > 0.7:
	#		print(cls, ' ', predVal)
	
	#print(IMAGE_NAME)
	with open('results.csv', 'w', newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		#writer.writerow(['Id'] + ['Class'])
		rootDir = os.path.normpath(ROOT_PATH) 
		#i = 0
		for i in range(0, 10000):
			#Loading image
	
			image = ndimage.imread(os.path.join(ROOT_PATH, str(i) + '_black_white_black_white.jpg'))
			image = image / 255.
		
			image = 1 - image
	
			image = image.reshape(1, 512, 256, 1)
			image = image.astype('float32')
	
			#Creating input to network from image
			feed_dict = {x: image}
	
			predictionArray = sess.run(y_pred, feed_dict)
				
			#print(os.path.join(ROOT_PATH, str(i) + '_black_white_black_white.jpg'))
			#for i in range(0, 26):
			#	print(classes[i], predictionArray[0][i])
			
			#print(file)
			#print(predictionArray)
				
			indexMax = predictionArray[0].argmax(axis=0)
			valueMax = predictionArray[0][indexMax]
			
			otherValue = predictionArray[0][25]
			
			if indexMax == 26:
				#print(indexMax)	
				writer.writerow(classes[indexMax])
			elif valueMax >= 0.8:
				#print(classes[indexMax])
				writer.writerow(classes[indexMax])
			elif otherValue > 1e-4:
				#print('Other')
				writer.writerow('Other')
			elif valueMax >= 0.6 and otherValue < 1e-5:
				#print(classes[indexMax])
				writer.writerow(classes[indexMax])
			else:
				#print('Other')
				writer.writerow('Other')
				
			#print('#############################')
