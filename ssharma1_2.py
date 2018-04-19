import os  
import sys  
stdout = sys.stderr 
sys.stderr = open(os.devnull,'w') 

import tensorflow as tf
import cv2
import os 
import numpy as np 
import time
import random
np.random.seed(7)

# ------------------ PREPROCESSING -------------------

directort_name=sys.argv[1]
image_size = 200

def images_array():
	index = 0
	size = (image_size, image_size)

	padColor = 0

	home = directort_name
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
        yield (x[i:i + n], y[i:i + n])

# ------------------ Creating Model -------------------
batch_size = 200
learning_rate = 0.001
max_steps = 1000
filter_size = 5
num_classes = 5	# Number of classes 
keep_rate = 0.8
num_epochs = 15

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
	print (shapeConv2)

	fc = tf.reshape(conv2, [-1, shapeConv2[1]*shapeConv2[2]*shapeConv2[3]])
	print (fc.get_shape().as_list())
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

def train_neural_network(x, y):
	prediction = cnn(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	train_op = tf.train.AdamOptimizer().minimize(cost)


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				
				# batch_x, batch_y = next_batch(train_x, train_y, batch_size)
				sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
		
				loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

train_neural_network(X, Y)
sys.stderr=Stdout

'''
with tf.variable_scope("conv1") as scope:
	kernel = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, num_features_conv1], stddev=0.1, dtype=tf.float32))
	biases = tf.Variable(tf.constant(0., shape=[num_features_conv1], dtype=tf.float32))
	conv_layer = tf.nn.conv2d(input=images, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
	pre_activation = tf.nn.bias_add(conv_layer, biases) #???: Is this the same as matmultiply and add
	conv1 = tf.nn.relu(pre_activation, scope.name) 

pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]) #TODO: play around with ksize and strides
norm1 = tf.nn.lrn(pool1, depth_radius=4, name="norm1")


with tf.variable_scope("conv2") as scope:
	kernel = tf.Variable(tf.truncated_normal(shape=[5, 5, num_features_conv1, num_features_conv2], stddev=0.1, dtype=tf.float32))
	biases = tf.Variable(tf.constant(0.1, shape=[num_features_conv2], dtype=tf.float32))
	conv_layer = tf.nn.conv2d(input=norm1, filter=kernel, strides=[1, 1, 1, 1], padding="SAME")
	pre_activation = tf.nn.bias_add(conv_layer, biases)
	conv2 = tf.nn.relu(pre_activation, scope.name)

#TODO: swap norm2 and pool2
norm2 = tf.nn.lrn(conv2, depth_radius=4, name="norm2") #???: Why is this before pool2
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1])

# Fully connected layer
with tf.variable_scope("fcl1") as scope:
	# Resize the matrix to make multiplication easier ie. flatten it 	
	reshape = tf.reshape(pool2, images.get_shape().as_list()[0], -1)
	dim = reshape.get_shape()[1].value

	weights = tf.Variable(tf.truncated_normal(shape=[dim, num_features_fcl1], stddev=0.4, dtype=tf.float32))
	biases = tf.Variable(tf.constant(0.01, shape=[num_features_fcl1], dtype=tf.float32))
	fcl1 = tf.relu(tf.matmul(reshape, weights)+biases, name=scope.name)

with tf.variable_scope("fcl2") as scope:
	weights = tf.Variable(tf.truncated_normal([num_features_fcl1, num_feature_fcl2], stddev=0.1, dtype=tf.float32))
	biases = tf.Variable(tf.constant(0.01, shape=[num_feature_fcl2]))
	fcl2 = tf.nn.relu(tf.matmul(fcl1, weights)+biases, name=scope.name)

with tf.variable_scope("softmax") as scope:
	weights = tf.Variable(tf.truncated_normal(shape=[num_feature_fcl2, num_classes], stddev=1/num_feature_fcl2))
	biases = tf.Variable(tf.constant(0.05, shape=[num_classes]))
	softmax = tf.add(tf.matmul((fcl2, weights)+biases, name=scope.name))

def loss(logits, labels):
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="cross_entropy")
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

  	# The total loss is defined as the cross entropy loss plus all of the weight
  	# decay terms (L2 loss).
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

# -------- Train -------- 

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for batch in range(batch_size):
		x_batch, y_batch = getData()

'''