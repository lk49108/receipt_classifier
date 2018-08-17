import tensorflow as tf
import constants as CONST
from ops import *
from spatial_transformer_network.transformer import *
import os
import cv2

# fully connected layer
fc_layer = lambda x, W, b, name=None: tf.nn.bias_add(tf.matmul(x, W), b)

class SGANModelWithSpatialTransformer():
	
	def __init__(self):
		self.d_bn1 = batch_norm(name='d_bn1')
		self.d_bn2 = batch_norm(name='d_bn2')
	
		self.g_bn0 = batch_norm(name='g_bn0')
		self.g_bn1 = batch_norm(name='g_bn1')
		self.g_bn2 = batch_norm(name='g_bn2')
			
		self.g_bn3 = batch_norm(name='g_bn3')

		self.build_model()
	
	def Flatten(self, layer):
		layer_shape = layer.get_shape()
		#num_features = tf.reduce_prod(tf.shape(layer)[1:])
		num_features = layer_shape[1:].num_elements()
		layer_flat = tf.reshape(layer, [-1, num_features])

		return layer_flat, num_features
	
	def lastLayer(self, x, wSize):
		name = 'last_layer_transformer'

		#1.0
		W1f = tf.get_variable(shape = [wSize, 16], initializer = tf.truncated_normal_initializer(stddev=0.01), name=name + 'd_w1f')
		b1f = tf.get_variable(shape = [16], initializer = tf.constant_initializer(value = 1.0), name = name + 'd_b1f')
	
		x1 = tf.matmul(x, W1f, name=name + 'd_el1f') + b1f
		x1 = tf.nn.relu(x1)
	
		W1 = tf.get_variable(shape = [16, 1], initializer = tf.truncated_normal_initializer(stddev=0.01), name=name + 'd_w1')
		b1 = tf.get_variable(shape = [1], initializer = tf.constant_initializer(value = 1.5), name = name + 'd_b1')
	
		el1 = tf.matmul(x1, W1, name=name + 'd_el1') + b1
		el1 = tf.nn.tanh(el1) * 0.15 + 0.85

		#0.0
		W2f = tf.get_variable(shape = [wSize, 1], initializer = tf.truncated_normal_initializer(stddev=0.01), name=name + 'd_w2f')
		#b2f = tf.get_variable(shape = [16], initializer = tf.constant_initializer(value = 1.0), name = name + 'd_b2f')
	
		x2 = tf.matmul(x, W2f, name=name + 'd_el1f') * 0
		#x2 = tf.nn.relu(x2)
	
		#W2 = tf.get_variable(shape = [16, 1], initializer = tf.truncated_normal_initializer(stddev=0.01), name=name + 'd_w2')
		#b2 = tf.get_variable(shape = [1], initializer = tf.constant_initializer(value = 0.0), name = name + 'd_b2')
	
		#el2 = tf.matmul(x2, W2, name=name + 'd_el2') + b2
	
		#0.0
		W3f = tf.get_variable(shape = [wSize, 16], initializer = tf.truncated_normal_initializer(stddev=0.01), name=name + 'd_w3f')
		b3f = tf.get_variable(shape=[16], initializer = tf.constant_initializer(value = 1.0), name = name + 'd_b3f')
	
		x3 = tf.matmul(x, W3f, name=name + 'd_el3f') + b3f
		x3 = tf.nn.relu(x3)
	
		W3 = tf.get_variable(shape = [16, 1], initializer = tf.truncated_normal_initializer(stddev=0.01), name=name + 'd_w3')
		b3 = tf.get_variable(shape = [1], initializer = tf.constant_initializer(value = 0.0), name = name + 'd_b3')
	
		el3 = tf.matmul(x3, W3, name=name + 'd_el3') + b3
		el3 + tf.nn.tanh(el3) * 0.2
        
		#0.0
		#W4f = tf.get_variable(shape = [wSize, 16], initializer = tf.truncated_normal_initializer(stddev=0.01), name=name + 'd_w4f')
		#b4f = tf.get_variable(shape=[16], initializer = tf.constant_initializer(value = 1.0), name = name + 'd_b4f')
	
		#x4 = tf.matmul(x, W4f, name=name + 'd_el4f') + b4f
		#x4 = tf.nn.relu(x4)
	
		#W4 = tf.get_variable(shape = [16, 1], initializer = tf.truncated_normal_initializer(stddev=0.01), name=name + 'd_w4')
		#b4 = tf.get_variable(shape = [1], initializer = tf.constant_initializer(value = 0.0), name = name + 'd_b4')
	
		#el4 = tf.matmul(x4, W4, name=name + 'd_el4') + b4
	
		#0.5
		W5f = tf.get_variable(shape = [wSize, 16], initializer = tf.truncated_normal_initializer(stddev=0.01), name=name + 'd_w5f')
		b5f = tf.get_variable(shape=[16], initializer = tf.constant_initializer(value = 1.0), name = name + 'd_b5f')
	
		x5 = tf.matmul(x, W5f, name=name + 'd_el5f') + b5f
		x5 = tf.nn.relu(x5)
	
		W5 = tf.get_variable(shape = [16, 1], initializer = tf.truncated_normal_initializer(stddev=0.01), name=name + 'd_w5')
		b5 = tf.get_variable(shape = [1], initializer = tf.constant_initializer(value = 0.0), name = name + 'd_b5')
	
		el5 = tf.matmul(x5, W5, name=name + 'd_el5') + b5
		el5 = tf.nn.tanh(el5) * 0.1 + 0.6
		
		#0.0
		W6f = tf.get_variable(shape = [wSize, 16], initializer = tf.truncated_normal_initializer(stddev=0.01), name=name + 'd_w6f')
		b6f = tf.get_variable(shape = [16], initializer = tf.constant_initializer(value = 1.0), name = name + 'd_b6f')
	
		x6 = tf.matmul(x, W6f, name=name + 'd_el6f') + b6f
		x6 = tf.nn.relu(x6)
	
		W6 = tf.get_variable(shape = [16, 1], initializer = tf.truncated_normal_initializer(stddev=0.01), name=name + 'd_w6')
		b6 = tf.get_variable(shape = [1], initializer = tf.constant_initializer(value = 0.43), name = name + 'd_b6')
	
		el6 = tf.matmul(x6, W6, name=name + 'd_el6') + b6
		el6 = tf.nn.tanh(el6) * 0.25 - 0.8
    
		#1 0const 0const 0const 1 0
		lastLayer = tf.concat([el1, x2, el3, x2, el5, el6], 1)
	
		return lastLayer
	
	def build_model(self):
		
		#input to network
		self.x = tf.placeholder(tf.float32, shape=[None, CONST.HEIGHT, CONST.WIDTH, CONST.IMAGE_DEPTH], name='x')
		self.z = tf.placeholder(tf.float32, shape=[None, CONST.z_dim], name='z')
		
		#build generator
		self.g_out = self.generator(self.z)
		#build discriminator
		self.d_out_real, self.d_fcn_real, self.transf_matrix_real = self.discriminator(self.x, reuseVariables = False)
		self.d_out_fake, self.d_fcn_fake, self.transf_matrix_fake = self.discriminator(self.g_out, reuseVariables = True)
		
		#cost calculations and optimizer for discriminator
		self.d_pred_cls_real = tf.argmax(self.d_out_real, dimension = 1)
	
		self.d_true_real = tf.placeholder(tf.float32, shape=[None, CONST.num_classes + 1], name='y_true_real')
		self.d_true_cls_real = tf.argmax(self.d_true_real, dimension=1)
	
		self.d_cross_entropy_real = tf.nn.softmax_cross_entropy_with_logits(logits=self.d_fcn_real, labels=self.d_true_real)
		self.d_cost_real = tf.reduce_mean(self.d_cross_entropy_real)
		
		#[0, 0, 0, ...., 1], dim = [?, NUM_CLASSES + 1]
		twenty_five_zeros = tf.zeros(shape = [CONST.batch_size, 25])
		one_one = tf.ones(shape = [CONST.batch_size, 1])
		self.d_true_fake = tf.concat([twenty_five_zeros, one_one], 1)
		
		print('shape_d_true_fake', self.d_true_fake.shape)
		
		self.d_cross_entropy_fake = tf.nn.softmax_cross_entropy_with_logits(logits=self.d_fcn_fake, labels=self.d_true_fake)
		self.d_cost_fake = tf.reduce_mean(self.d_cross_entropy_fake)
		
		self.d_cost = self.d_cost_fake + self.d_cost_real
		
		self.d_correct_prediction_real = tf.equal(self.d_pred_cls_real, self.d_true_cls_real)
		self.d_accuracy_real = tf.reduce_mean(tf.cast(self.d_correct_prediction_real, tf.float32))
		
		#cost calculations and optimizer for generator
		
		#[1, 1, 1, ...., 0], dim = [?, NUM_CLASSES + 1]
		twenty_five_ones = tf.ones(shape = [CONST.batch_size, 25])
		one_zero = tf.zeros(shape = [CONST.batch_size, 1])
		self.g_labels = tf.concat([twenty_five_ones, one_zero], 1)
		
		print('vector1110', self.g_labels.shape)
		
		self.g_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.d_fcn_fake, labels=self.g_labels)
		self.g_cost = tf.reduce_mean(self.g_cross_entropy)
		
		#g_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(g_cost)		
	
		#storing discriminator and generator variables
		t_vars = tf.trainable_variables()
		
		self.d_vars = [var for var in t_vars if 'd_' in var.name]
		self.g_vars = [var for var in t_vars if 'g_' in var.name]
	
	def train(self, trainingData, num_iteration = 1000000):
		d_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.d_cost, var_list = self.d_vars)
		g_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.g_cost, var_list = self.g_vars)
		
		#set this if you want to start training from checkpoint
		#also set the right checkpoint folder
		continueFromCheckpoint = True
		CHECKPOINT_FOLDER = './models'
		
		with tf.Session() as session:
			saver = tf.train.Saver(max_to_keep=10000)
			
			if not continueFromCheckpoint:
				session.run(tf.global_variables_initializer()) 			
			else:
				saver.restore(session, tf.train.latest_checkpoint(CHECKPOINT_FOLDER))
				print("restoring...")
				
			def show_progress(iteration, trainingAccuracy, curDPercentageCorrect, curDRealCost, curDFakeCost):
				msg = "Training iteration {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Real Loss: {3:.3f},  Validation Fake Loss: {4:.3f}"
				print(msg.format(iteration, trainingAccuracy, curDPercentageCorrect, curDRealCost, curDFakeCost))
		
			modelsDir = os.path.join(os.path.normpath('.'), os.path.normpath('models'))
	
			if not os.path.exists(modelsDir):
				os.makedirs(modelsDir)
			
			saver.save(session, str(modelsDir) + '/leonardo_model-' + str(0))
		
			#also set right i
			start_point = 191000 #### zapoceti od zadnje nastale iteracije + 1000 u mapi c/solution/models
			for i in range(0 + start_point, num_iteration + start_point):
				#creating z vector
				z_batch_vector = np.random.uniform(-1, 1, size=[CONST.batch_size, CONST.z_dim]).astype(np.float32)
		
				#update discriminator
				x_batch, y_true_batch, _, cls_batch = trainingData.train.next_batch(CONST.batch_size)
				d_tr_dict = {
						self.x : x_batch,
						self.z : z_batch_vector,
						self.d_true_real : y_true_batch
					}
			
				_, curDCost = session.run([d_optimizer, self.d_cost], feed_dict = d_tr_dict)
				#print('d_cost', curDCost[0])
			
				#matrix = session.run(self.transf_matrix_real, feed_dict = d_tr_dict)
				#print(matrix[0])
				#spa = session.run(self.spatial_trans, feed_dict = d_tr_dict)
				#cv2.imshow("image", spa[0])
				#cv2.waitKey(0)
				
				#update generator
				_, curGCost = session.run([g_optimizer, self.g_cost], feed_dict = {self.z : z_batch_vector})
				#print('g_cost_1', curGCost[0])
				#update generator once more :D
				_, curGCost = session.run([g_optimizer, self.g_cost], feed_dict = {self.z : z_batch_vector})
				#print('g_cost_2', curGCost[0])
			
				#curDRealCost = session.run(self.d_cost_real, feed_dict = {self.x : x_batch, self.d_true_real : y_true_batch}) 
				#curDFakeCost = session.run(self.d_cost_fake, feed_dict = {self.z : z_batch_vector})
			
				#curDPercentageCorrect = session.run(self.d_accuracy_real, feed_dict = {self.x : x_batch, self.d_true_real : y_true_batch})
			
				if i % 1 == 0:
					x_batch_val, y_true_batch_val, _, cls_batch_val = trainingData.valid.next_batch(10)
				
					trainingAccuracy = session.run(self.d_accuracy_real, feed_dict = {self.x : x_batch, self.d_true_real : y_true_batch})
					curDPercentageCorrect = session.run(self.d_accuracy_real, feed_dict = {self.x : x_batch_val, self.d_true_real : y_true_batch_val})
					curDRealCost = session.run(self.d_cost_real, feed_dict = {self.x : x_batch_val, self.d_true_real : y_true_batch_val}) 
					curDFakeCost = session.run(self.d_cost_fake, feed_dict = {self.z : z_batch_vector})
				
					matrix = session.run(self.transf_matrix_real, feed_dict = {self.x : x_batch_val})
					print(matrix[0])
				
					show_progress(i, trainingAccuracy, curDPercentageCorrect, curDRealCost, curDFakeCost)
				
					saver.save(session, str(modelsDir) + '/leonardo_model-' + str(i))
				
	def discriminator(self, x, reuseVariables = False):
		with tf.variable_scope("discriminator") as scope:
			if reuseVariables:
				scope.reuse_variables()
				
			spatial_trans, transform_matrix = self.spatialTransformer(x)
	
			if not reuseVariables:
				self.spatial_trans = spatial_trans
	
			y_pred, y_fcn = self.cnn(spatial_trans)
		
			return y_pred, y_fcn, transform_matrix
		
	def spatialTransformer(self, x, name = 'spatial_transformer'):
		# Weight parameters as devised in the original research paper
		weights = {
			"wc1": tf.get_variable(shape = [11, 11, CONST.IMAGE_DEPTH, 16], initializer = tf.truncated_normal_initializer(stddev=0.01), name="d_trans_wc1"),
			"wc2": tf.get_variable(shape = [5, 5, 16, 32], initializer = tf.truncated_normal_initializer(stddev=0.01), name="d_trans_wc2"),
			"wf1": tf.get_variable(shape = [128, 32], initializer = tf.truncated_normal_initializer(stddev=0.01), name="d_trans_wf1")
		}

		# Bias parameters as devised in the original research paper
		biases = {
			"bc1": tf.get_variable(shape = [16], initializer = tf.constant_initializer(value = 1.0), name="d_trans_bc1"),
			"bc2": tf.get_variable(shape=[32], initializer = tf.constant_initializer(value = 1.0), name="d_trans_bc2"),
			"bf1": tf.get_variable(shape=[32], initializer = tf.constant_initializer(value = 1.0), name="d_trans_bf1")
		}
		
		# 1st convolutional layer
		conv1 = tf.nn.conv2d(x, weights["wc1"], strides=[1, 5, 5, 1], padding="VALID", name="d_trans_conv1")
		conv1 = tf.nn.bias_add(conv1, biases["bc1"])
		conv1 = tf.nn.relu(conv1)
		conv1 = tf.nn.local_response_normalization(conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
	
		maxPool1 = tf.nn.max_pool(conv1, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="VALID")

		# 2nd convolutional layer
		conv2 = tf.nn.conv2d(maxPool1, weights["wc2"], strides=[1, 3, 3, 1], padding="SAME", name="d_trans_conv2")
		conv2 = tf.nn.bias_add(conv2, biases["bc2"])
		conv2 = tf.nn.relu(conv2)
		conv2 = tf.nn.local_response_normalization(conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
	
		maxPool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
	
		flatten, flatten_dim = self.Flatten(maxPool2)
		
		# 1st fully connected layer
		fc1 = fc_layer(flatten, weights["wf1"], biases["bf1"], name="d_trans_fc1")
		fc1 = tf.nn.relu(fc1)
		fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

		#2nd fully connected layer
		fcn2 = self.lastLayer(fc1, wSize = 32)
	
		return spatial_transformer_network(x, fcn2, CONST.CROP_SIZE), fcn2
	
	def cnn(self, x):
		weights = {
			"wc1": tf.get_variable(shape = [11, 11, CONST.IMAGE_DEPTH, 32], initializer = tf.truncated_normal_initializer(stddev=0.01), name="d_wc1"),
			"wc2": tf.get_variable(shape = [4, 4, 32, 48], initializer = tf.truncated_normal_initializer(stddev=0.01), name="d_wc2"),
			"wc3": tf.get_variable(shape = [3, 3, 48, 64], initializer = tf.truncated_normal_initializer(stddev=0.01), name="d_wc3"),
			"wf1": tf.get_variable(shape = [256, 32], initializer = tf.truncated_normal_initializer(stddev=0.01), name="d_wf1"),
			"wf2": tf.get_variable(shape = [32, CONST.num_classes + 1], initializer = tf.truncated_normal_initializer(stddev=0.01), name="d_wf2"),
		}

		# Bias parameters as devised in the original research paper
		biases = {
			"bc1": tf.get_variable(shape = [32], initializer = tf.constant_initializer(value = 1.0), name="d_bc1"),
			"bc2": tf.get_variable(shape = [48], initializer = tf.constant_initializer(value = 1.0), name="d_bc2"),
			"bc3": tf.get_variable(shape = [64], initializer = tf.constant_initializer(value = 1.0), name="d_bc3"),
			"bf1": tf.get_variable(shape = [32], initializer = tf.constant_initializer(value = 1.0), name="d_bf1"),
			"bf2": tf.get_variable(shape = [CONST.num_classes + 1], initializer = tf.constant_initializer(value = 1.0), name="d_bf2"),
		}

		# 1st convolutional layer
		conv1 = tf.nn.conv2d(x, weights["wc1"], strides=[1, 5, 5, 1], padding="SAME", name="d_conv1")
		conv1 = tf.nn.bias_add(conv1, biases["bc1"])
		conv1 = tf.nn.relu(conv1)
		conv1 = tf.nn.local_response_normalization(conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
		conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="VALID")

		# 2nd convolutional layer
		conv2 = tf.nn.conv2d(conv1, weights["wc2"], strides=[1, 3, 3, 1], padding="SAME", name="d_conv2")
		conv2 = tf.nn.bias_add(conv2, biases["bc2"])
		conv2 = tf.nn.relu(conv2)
		conv2 = tf.nn.local_response_normalization(conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
		conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
	
		# 3rd convolutional layer
		conv3 = tf.nn.conv2d(conv2, weights["wc3"], strides=[1, 2, 2, 1], padding="SAME", name="d_conv3")
		conv3 = tf.nn.bias_add(conv3, biases["bc3"])
		conv3 = tf.nn.relu(conv3)

		# stretching out the 3rd convolutional layer into a long n-dimensional tensor
		flatten, flatten_size = self.Flatten(conv3)
	
		# 1st fully connected layer
		fc1 = fc_layer(flatten, weights["wf1"], biases["bf1"], name="d_fc1")
		fc1 = tf.nn.tanh(fc1)
		fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

		# 2nd fully connected layer
		fc2 = fc_layer(fc1, weights["wf2"], biases["bf2"], name="d_fc2")
	
		y_pred = tf.nn.softmax(fc2, name="d_y_pred")
		
		return y_pred, fc2
		
	def generator(self, z):
		with tf.variable_scope('generator') as scope:
			s_h, s_w = CONST.HEIGHT, CONST.WIDTH
			s_h2, s_w2 = int(s_h / 2), int(s_w / 2)
			s_h4, s_w4 = int(s_h2 / 2), int(s_w2 / 2)
			s_h8, s_w8 = int(s_h4 / 2), int(s_w4 / 2)
			s_h16, s_w16 = int(s_h8 / 2), int(s_w8 / 2)

			# project `z` and reshape, change constant if necessary
			z_ = linear(
				z, 256*s_h16*s_w16, 'g_h0_lin')

			h0 = tf.reshape(
				z_, [-1, s_h16, s_w16, 256])
			h0 = tf.nn.relu(self.g_bn0(h0))

			#h1 = deconv2d(
			#	h0, [CONST.batch_size, s_h8, s_w8, 128], name='g_h1')
			#h1 = tf.nn.relu(self.g_bn1(h1))

			#h2 = deconv2d(
			#	h1, [CONST.batch_size, s_h16, s_w16, 256], name='g_h2')
			#h2 = tf.nn.relu(self.g_bn2(h2))

			h3 = deconv2d(
				h0, [CONST.batch_size, s_h4, s_w4, 64], name='g_h3')
			h3 = tf.nn.relu(self.g_bn3(h3))

			h4 = deconv2d(
				h3, [CONST.batch_size, s_h, s_w, CONST.IMAGE_DEPTH], name='g_h4')

			return tf.nn.tanh(h4)