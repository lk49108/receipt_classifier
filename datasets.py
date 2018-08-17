import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
from scipy import ndimage
import constants as CONST

def loadTrainingData(train_path, classes):
	#images = []
	labels = []
	imgNames = []
	cls = []

	print('Going to read training images')
	for subdir, dirs, files in os.walk(train_path):
		if len(subdir.split('\\')) == 1:
			continue
		
		className = subdir.split('\\')[1]
		print(className)
		
		for file in files:
			filePath = os.path.join(subdir, file)

			#Reading and processing image
			#image = cv2.imread(filePath)
			#image = np.multiply(image, 1.0 / 255.0)
			#image = np.around(image)
			#images.append(image)
			
			#Creating one-hot-shot array for image classification label
			label = np.zeros(len(classes) + 1)
			
			index = classes.index(className)
			
			label[index] = 1.0
			labels.append(label)
			
			#Creating filename
			imgNames.append(filePath)
			
			#Appending classname
			cls.append(className)
			
			#print(tf.reshape(image, [-1, 1052, 564, 3]))
	
	#images = np.array(images)
	labels = np.array(labels)
	imgNames = np.array(imgNames)
	cls = np.array(cls)
	
	return labels, imgNames, cls

class DataSet(object):

	def __init__(self, labels, img_names, cls):
		self._num_examples = img_names.shape[0]

		self._labels = labels
		self._img_names = img_names
		self._cls = cls
		self._epochs_done = 0
		self._index_in_epoch = 0


	@property
	def labels(self):
		return self._labels

	@property
	def img_names(self):
		return self._img_names

	@property
	def cls(self):
		return self._cls

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_done(self):
		return self._epochs_done

	def next_batch(self, batch_size):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size

		if self._index_in_epoch > self._num_examples:
			# After each epoch we update this
			self._epochs_done += 1
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		
		end = self._index_in_epoch

		#return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]
		return self.loadImages(start, end), self._labels[start:end], self._img_names[start:end], self._cls[start:end]
	
	def readImage(self, index):
		#Reading and processing image
		image = ndimage.imread(self._img_names[index])
		#image = cv2.resize(image, (CONST.WIDTH, CONST.HEIGHT))
		image = image / 255.
		
		image = 1 - image
		
		#cv2.imshow('image', image)
		#cv2.waitKey(0)
		image = image.reshape(1, CONST.HEIGHT, CONST.WIDTH, 1)
		image = image.astype('float32')
			
		return image
	
	def loadImages(self, start, end):
		
		imageOne = self.readImage(start)
		batch = None
		
		for i in range(start+1, end):
			if i == start+1:
				batch = np.append(imageOne, self.readImage(i), axis=0)
			else:
				batch = np.append(batch, self.readImage(i), axis=0)
	
		return batch
		
def loadTrainingSet(train_root_path, classes, validation_size):
	class DataSets(object):
		pass
	
	data_sets = DataSets()

	#images, labels, img_names, cls = loadTrainingData(train_root_path, classes)
	#images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  
	labels, img_names, cls = loadTrainingData(train_root_path, classes)
	labels, img_names, cls = shuffle(labels, img_names, cls)  
	
	if isinstance(validation_size, float):
		validation_size = int(validation_size * img_names.shape[0])

	#validation_images = images[:validation_size]
	validation_labels = labels[:validation_size]
	validation_img_names = img_names[:validation_size]
	validation_cls = cls[:validation_size]

	#train_images = images[validation_size:]
	train_labels = labels[validation_size:]
	train_img_names = img_names[validation_size:]
	train_cls = cls[validation_size:]

	#data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
	#data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

	data_sets.train = DataSet(train_labels, train_img_names, train_cls)
	data_sets.valid = DataSet(validation_labels, validation_img_names, validation_cls)
	
	return data_sets