import os
import numpy as np
import math

from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist



# Generator (G)
def generator():
	""" <I
	The last layer needs to be formated to output
	an image in the same format as the formated images
	from the dataset. 

	Also: G.lastlayer = D.firstlayer
	"""
	model = Sequential()
	# 1*1*8192 (flat input conv with batch_size=BATCH_SIZE=5 und input_dim=100?)
	# 4*4*512
	# 8*8*256
	# 16*16*128
	# 32*32*64
	# 64*64*32
	# output conv 128*128*3


	"""
	model.add(Dense(units=1024, input_dim=100))
	model.add(Activation('tanh'))
	model.add(Dense(128*7*7))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(64, (5, 5), padding='same'))
	model.add(Activation('tanh'))
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(1, (5, 5), padding='same'))
	model.add(Activation('tanh'))
	"""
	return model


# Discriminator (D)
def discriminator():
	""" I>
	"""	
	#model = Sequential()
	# 128*128*3
	# 64*64*32
	# 32*32*64
	# 16*16*128
	# 8*8*256
	# 4*4*512
	# 1*1*8192?d oder 2 (binary  classification)???





		
	"""
	model = Sequential()
	model.add(Conv2D(64, (5, 5), padding='same', input_shape=(128, 128, 1)))
	model.add(Activation('tanh'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (5, 5)))
	model.add(Activation('tanh'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation('tanh'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	return model
	"""


# Combine the batch of generated uint8 arrays into an image composition
def combine_images(generated_images):
	""" 
	generated_images: (Input) A batch of images as an uint8 array.
	"""
	num = generated_images.shape[0]
	width = int(math.sqrt(num))
	height = int(math.ceil(float(num) / width))
	shape = generated_images.shape[1:3]
	image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)

	for index, img in enumerate(generated_images):
		i = int(index / width)
		j = index % width
		image[i * shape[0]:(i+1) * shape[0], j * shape[1]:(j+1) * shape[1]] = \
				img[:, :, 0]
	return image


# Train the GAN
def train(EPOCHS, BATCH_SIZE, SAVE_INTERVAL, IMAGE_SIZE):
	"""
	EPOCHS: The number of training iterations.
	BATCH_SIZE: The number 
	SAVE_INTERVAL: How many epochs to wait before saving a generated sample.
	IMAGE_SIZE: The input and output images will be scaled to this size.
	"""

	# Create the GAN for this mode
	model = Sequential()
	g = generator()
	d = discriminator()
	model.add(g)
	model.add(d)

	d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
	g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

	g.compile(loss='binary_crossentropy', optimizer="SGD")
	model.compile(loss='binary_crossentropy', optimizer=g_optim)
	d.trainable = True
	d.compile(loss='binary_crossentropy', optimizer=d_optim)


	# Create (augmented) data batch iterator
	train_batch = ImageDataGenerator()
	train_batch = train_batch.flow_from_directory(
		'data',
		target_size=(IMAGE_SIZE, IMAGE_SIZE),
		color_mode="rgb",
		class_mode="input",
		batch_size=BATCH_SIZE,
		shuffle=True,
		seed=42)

	train_batch = next(train_batch)


	# The actual algorithm as described in the original paper with k=1:
	# =================================================================
	for epoch in range(EPOCHS):
		print("Epoch: ", epoch)

		# 1. Sample minibatch of m noise samples from noise prior
		noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

		# 2. Sample minibatch of m examples from data generating distribution
		generated_images = g.predict(noise, verbose=0)

		# (save generated uint8 arrays as .png images)
		image = combine_images(generated_images)
		image = image*127.5+127.5 
		Image.fromarray(image.astype(np.uint8)).save(str(epoch)+".png")

		# 3. Update the discriminator D by ascending its stochastic greadient
		X = np.concatenate((train_batch, generated_images))
		y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
		d_loss = d.train_on_batch(X, y)
		print("batch %d d_loss : %f" % (index, d_loss))

		# 4. Sample minibatch of m noise samples
		noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

		# 5. Update the generator G by descenging its stochastic gradient
		d.trainable = False
		g_loss = model.train_on_batch(noise, [1] * BATCH_SIZE)
		d.trainable = True
		print("batch %d g_loss : %f" % (index, g_loss))

		# (save weights)
		g.save_weights('generator', True)
		d.save_weights('discriminator', True)


# Load the trained model and generate a collection of images
def generate(SAMPLES):
	"""
	"""
	pass


# argparse functionality for simplicity
# train the GAN (the weights will be saved). 
# The net gets automaticly trained on the pictures in the data folder.
train(EPOCHS=13, BATCH_SIZE=100, SAVE_INTERVAL=5, IMAGE_SIZE=128)

# Generate new images with the trained GAN
#generate(SAMPLES=1)