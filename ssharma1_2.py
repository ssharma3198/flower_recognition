import os  
import sys  

import tensorflow as tf
import cv2
import os 
import numpy as np 
import time
import random
np.random.seed(7)

# ------------------ PREPROCESSING -------------------

# directort_name=sys.argv[1]
image_size = 200

def images_array():
	index = 0
	size = (image_size, image_size)

	padColor = 0

	home = "/Users/Shruti/Desktop/School/Spring 2018/PHYS 476/Homework/HW 3/2. CNN/flowers/"
	classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
	data = []
	for c in classes:
		one_hot = np.zeros(shape=5)
		one_hot[index] = 1
		path = home + c
		images = os.listdir(path)
		for image in images: 
			if (image[-4:] == ".jpg"):
				# resize image
				flower = path + "/" + image
				img = cv2.imread(flower)
				h, w = img.shape[:2]
				sh, sw = size

			    # interpolation method
				if h > sh or w > sw:
			    	# shrinking image
					interp = cv2.INTER_AREA
				else: 
			    	# stretching image
					interp = cv2.INTER_CUBIC

			    # aspect ratio of image
				aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

			    # compute scaling and pad sizing
				if aspect > 1: # horizontal image
					new_w = sw
					new_h = np.round(new_w/aspect).astype(int)
					pad_vert = (sh-new_h)/2
					pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
					pad_left, pad_right = 0, 0
				elif aspect < 1: # vertical image
					new_h = sh
					new_w = np.round(new_h*aspect).astype(int)
					pad_horz = (sw-new_w)/2
					pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
					pad_top, pad_bot = 0, 0
				else: # square image
					new_h, new_w = sh, sw
					pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

			    # set pad color
				if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
					padColor = [padColor]*3

			    # scale and pad
				scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
				scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

				data.append([scaled_img, one_hot])
		index += 1
	return data 

def create_data():
	test_ratio = 0.25
	data = images_array()
	random.shuffle(data)

	features = np.array(data)
	testing_size = int(test_ratio*len(data))

	train_x = np.array(list(features[:,0][:-testing_size]))
	train_y = np.array(list(features[:,1][:-testing_size]))
	test_x = np.array(list(features[:,0][-testing_size:]))
	test_y = np.array(list(features[:,1][-testing_size:])) 

	return train_x,train_y,test_x,test_y


def next_batch(x, y, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(x), n):
        yield x[i:i + n], y[i:i + n]

# ------------------ Creating Model -------------------

batch_size = 200
learning_rate = 0.001
max_steps = 1000
filter_size = 5
num_classes = 5	# Number of classes 
keep_rate = 0.8
num_epochs = 15
num_images = 4324

num_features_conv1 = 128
num_features_conv2 = 256
num_features_conv3 = 256
num_features_fcl = 512
num_input = image_size*image_size

X = tf.placeholder("float", [None, image_size, image_size, 3])
Y = tf.placeholder("float", [None, num_classes])

def conv_layer(input, weights, biases):
	layer = tf.nn.conv2d(input, weights,  strides=[1,1,1,1], padding='SAME')
	layer = tf.nn.bias_add(layer, biases)
	layer = tf.nn.relu(layer)
	return tf.nn.max_pool(layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def cnn(x):
	weights = {'W_conv1':tf.Variable(tf.random_normal([7,7,3,num_features_conv1])), # 5x5 feature, 1 input num_feautres_conv1 putputs
				'W_conv2':tf.Variable(tf.random_normal([5,5,num_features_conv1,num_features_conv2])),
				'W_fc':tf.Variable(tf.random_normal([50*50*num_features_conv2,num_features_fcl])), # 7x7 final pooled?
				'out':tf.Variable(tf.random_normal([num_features_fcl, num_classes]))}

	biases = {'b_conv1':tf.Variable(tf.random_normal([num_features_conv1])),
				'b_conv2':tf.Variable(tf.random_normal([num_features_conv2])),
				'b_fc':tf.Variable(tf.random_normal([num_features_fcl])),
				'out':tf.Variable(tf.random_normal([num_classes]))}


	conv1 = conv_layer(x, weights['W_conv1'], biases['b_conv1'])
	conv2 = conv_layer(conv1, weights['W_conv2'], biases['b_conv2'])

	shapeConv2 = conv2.get_shape().as_list()

	fc = tf.reshape(conv2, [-1, shapeConv2[1]*shapeConv2[2]*shapeConv2[3]])

	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights['out']) + biases['out']
	
	return output
	'''
	weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,num_features_conv1])), # 5x5 feature, 1 input num_feautres_conv1 putputs
				'W_conv2':tf.Variable(tf.random_normal([5,5,num_features_conv1,num_features_conv2])),
				'W_conv3':tf.Variable(tf.random_normal([5, 5, num_features_conv2, num_features_conv3])),
				'W_fc':tf.Variable(tf.random_normal([4*4*num_features_conv2,num_features_fcl])), # 7x7 final pooled?
				'out':tf.Variable(tf.random_normal([num_features_fcl, num_classes]))}

	biases = {'b_conv1':tf.Variable(tf.random_normal([num_features_conv1])),
				'b_conv2':tf.Variable(tf.random_normal([num_features_conv2])),
				'b_conv3':tf.Variable(tf.random_normal([num_features_conv3])),
				'b_fc':tf.Variable(tf.random_normal([num_features_fcl])),
				'out':tf.Variable(tf.random_normal([num_classes]))}

	conv1 = conv_layer(x, weights['W_conv1'], biases['b_conv1'])
	conv2 = conv_layer(conv1, weights['W_conv2'], biases['b_conv2'])
	conv3 = conv_layer(conv2, weights['W_conv3'], biases['b_conv3'])

	shapeConv3 = conv3.get_shape().as_list()
	print (shapeConv3)

	fc = tf.reshape(conv3, [-1, shapeConv3[1]*shapeConv3[2]*shapeConv3[3]])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights['out']) + biases['out']
	
	return output

	'''
	
	

# --------------------- Testing --------------------

train_x,train_y,test_x,test_y = create_data()
num_batches = int(len(train_x) / batch_size)
batches = next_batch(train_x, train_y, batch_size)

def train_neural_network(x, y):
	prediction = cnn(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	train_op = tf.train.AdamOptimizer().minimize(cost)

	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):
			for batch in range(num_batches):
				batch_x, batch_y = next(batches)
				sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
			print ("Accuracy of epoch %d %d" , epoch, sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))
			
		print ("Accuracy: ", sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))
train_neural_network(X, Y)
sys.stderr=Stdout
