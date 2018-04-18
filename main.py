import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, AveragePooling2D
from keras.layers import BatchNormalization, UpSampling2D, Conv2D
from keras.layers import Activation, Dropout, LeakyReLU, Conv2DTranspose
from keras.layers import GaussianNoise, GaussianDropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import movie
import parameters


# generator model
def createGenerator():

	g = Sequential(name="Generator")
	
	g.add(Dense(768, kernel_initializer=dense_kernel_init, input_shape=(Z,)))
	g.add(Activation("relu"))
	g.add(Reshape((1, 1, 768)))
	
	g.add(UpSampling2D(size=(2, 2)))
	g.add(Conv2D(192, 4, kernel_initializer=conv_kernel_init, padding='same'))
	g.add(BatchNormalization(axis=-1, momentum=0.1))
	g.add(Activation('relu'))

	g.add(UpSampling2D(size=(2, 2)))
	g.add(Conv2D(48, 4, kernel_initializer=conv_kernel_init, padding='same'))
	g.add(BatchNormalization(axis=-1, momentum=0.1))
	g.add(Activation('relu'))

	g.add(UpSampling2D(size=(2, 2)))
	g.add(Conv2D(12, 4, kernel_initializer=conv_kernel_init, padding='same'))
	g.add(BatchNormalization(axis=-1, momentum=0.1))
	g.add(Activation('relu'))

	g.add(UpSampling2D(size=(2, 2)))
	g.add(Conv2D(3, 4, kernel_initializer=conv_kernel_init, padding='same'))
	g.add(Activation('tanh'))

	print(g.summary())

	return g

# discriminator model
def createDiscriminator():

	d = Sequential(name="Discriminator")

	d.add(GaussianNoise(D_input_noise, input_shape=(16, 16, 3)))

	d.add(Conv2D(12, 5, kernel_initializer=conv_kernel_init, padding='same'))
	d.add(LeakyReLU(alpha=0.2))
	d.add(AveragePooling2D(pool_size=(2, 2), strides=2))

	d.add(GaussianDropout(D_dropout))

	d.add(Conv2D(48, 5, kernel_initializer=conv_kernel_init, padding='same'))
	#d.add(BatchNormalization(axis=-1))
	d.add(LeakyReLU(alpha=0.2))
	d.add(AveragePooling2D(pool_size=(2, 2), strides=2))

	d.add(GaussianDropout(D_dropout))

	d.add(Conv2D(192, 5, kernel_initializer=conv_kernel_init, padding='same'))
	#d.add(BatchNormalization(axis=-1))
	d.add(LeakyReLU(alpha=0.2))
	d.add(AveragePooling2D(pool_size=(2, 2), strides=2))
	
	d.add(GaussianDropout(D_dropout))

	d.add(Conv2D(768, 5, kernel_initializer=conv_kernel_init, padding='same'))
	#d.add(BatchNormalization(axis=-1))
	d.add(LeakyReLU(alpha=0.2))
	d.add(AveragePooling2D(pool_size=(2, 2), strides=2))

	d.add(GaussianDropout(D_dropout))

	# Classifier
	d.add(Flatten())

	d.add(Dense(1, kernel_initializer=dense_kernel_init))
	d.add(Activation("sigmoid"))

	print(d.summary())

	return d

x = []; X = []
y = []; Y = []

# live plot the training V2
def plotVis(epoch, d_loss, g_loss):
	x.append(d_loss[0])
	y.append(g_loss[0])
	X.append(d_loss[1])
	Y.append(g_loss[1])
	global fig, ax0, ax1
	ax0.legend(loc='upper center', 
		bbox_to_anchor=(0.5, 1.15), 
		fancybox=False, shadow=True, ncol=2)
	ax0.plot(y, '#50964B')
	ax0.plot(x, '#B1B1B1') 
	ax1.plot(Y, '#50964B')
	ax1.plot(X, '#B1B1B1') 
	plt.pause(0.1)
	plt.show(block=False)
   	plt.pause(0.1)

# label smoothing
def smooth(label, STRENGTH=0.1):
	label = label.astype(float)
	pos = 0
	for e in np.nditer(label, op_flags=['readwrite']):
		if e == 1:
			e  = e - (STRENGTH * np.random.random_sample())
		elif e  == 0:
			e  = e + (STRENGTH * np.random.random_sample())
		np.put(label, [pos], [e])	
		pos += 1	
	return label

# train the GAN
def train(BATCH_SIZE, EPOCHS, D, G, S, data_path, G_optimizer, D_optimizer):

	# init train history.txt file
	file = open('history/history.txt', 'w') 

	# init the GAN
	gan = Sequential()
	g = createGenerator()
	d = createDiscriminator()
	d.trainable = False
	gan.add(g)
	gan.add(d)

	g.compile(loss='binary_crossentropy', 
		optimizer=G_optimizer)
	gan.compile(loss='binary_crossentropy', 
		optimizer=D_optimizer,
		metrics=['accuracy'])
	d.trainable = True
	d.compile(loss='binary_crossentropy', 
		optimizer=D_optimizer,
		metrics=['accuracy'])

	if load_weights == 1:
		g.load_weights('history/generator.h5')
		d.load_weights('history/discriminator.h5')


	# augmentation configuration for training
	real_batch = ImageDataGenerator(
		rescale=1. / 255, 
		shear_range=0.0, 
		zoom_range=0.0, 
		horizontal_flip=True,
		data_format="channels_last")
	# enable iterating through the directory
	real_batch = real_batch.flow_from_directory(
		("data/" + data_path), 
		target_size=(16, 16),
		color_mode="rgb",	
		batch_size=BATCH_SIZE,	
		class_mode=None,
		shuffle=True)

	real_batch = next(real_batch)
	real_batch = np.array(real_batch)

	#d_loss_mem = 0.5
	#g_loss_mem = 0.5

	# train
	for epoch in range(EPOCHS):
		
		print("=================================================================")
		print("Epoch "+ str(epoch)) + "/" + str(EPOCHS)+ " | D = " + str(D) + " | G = " + str(G)
		print("_________________________________________________________________")


		for step in range(D):
			# 1. Sample minibatch of m noise samples from noise prior
			noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z)) 
			# 2. Sample minibatch of m examples from data generating distribution
			fake_batch = np.array(g.predict(noise, verbose=0))
			# 3. Update the discriminator D by ascending its stochastic greadient
			data = np.concatenate((real_batch, fake_batch))
			label = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
			label = smooth(label, S)
			d.trainable = True
			d_loss = d.train_on_batch(x=data, y=label)
			#d_loss_mem = d_loss[0]
			print("Discriminator | loss: " + str(d_loss[0]) + ", acc.: " + str(d_loss[1]))			

		for step in range(G):
			# 4. Sample minibatch of m noise samples (redundant for k=1)
			noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, Z))
			# 5. Update the generator G by descenging its stochastic gradient
			d.trainable = False
			g_loss = gan.train_on_batch(x=noise, y=np.array([1] * BATCH_SIZE))
			print("Generator     | loss: " + str(g_loss[0]) + ", acc.: " + str(g_loss[1]))

		
		# save weights as .h5 file
		if epoch % 1 == 0:
			g.save_weights('history/generator.h5', True)
			d.save_weights('history/discriminator.h5', True)
		# save losses as .txt file
		file = open('history/history.txt', 'a+') 
		file.write(str(d_loss) + "," + str(g_loss) + '\n')
		# select are random image array out of the complete array of generated images 
		pic = (np.vsplit(fake_batch, BATCH_SIZE))[np.random.randint(BATCH_SIZE)]
		# reformat image array
		pic = pic*255
		pic = np.squeeze(pic, axis=0)
		pic = pic.astype(np.uint8)
		# save image array as jpg in diretory
		Image.fromarray(pic, "RGB").save("gen/Img_"+str(epoch)+".jpg")
		print("...sample image saved as "+"Img_"+str(epoch)+".jpg")

		# show a live plot
		plotVis(epoch, d_loss, g_loss)
	

	# create video from all the saved images
	movie.createVideo()

	# save plot
	plt.savefig("history/history"+".jpg", dpi='figure', facecolor='w', edgecolor='w', 
		orientation='portrait', papertype=None, format=None, 
		transparent=False, bbox_inches=None, pad_inches=0.1,
		frameon=None)
	# hold program and show plot
	plt.show(block=True)


# generate and save images
def generate(BATCH_SIZE):
	# compile generator
	g = createGenerator()
	g.compile(loss='binary_crossentropy', optimizer="SGD")
	# load weights of the generator
	g.load_weights('history/generator.h5')
	# save each image individually to the cdr
	for sample in range(BATCH_SIZE):
		# create noise for generator
		noise = np.random.uniform(-1, 1, (1, Z))
		# create array of a set of images
		generated_image = g.predict(noise, verbose=1)
		# reformat image array
		pic = generated_image*255
		pic = np.squeeze(pic, axis=0)
		pic = pic.astype(np.uint8)
		# save image array as jpg in diretory
		Image.fromarray(pic, "RGB").save("Img_"+str(sample+1)+".jpg")

# execute program with given parameters
if __name__ == "__main__":

	# set training parameters
	np.random.seed(parameters.SEED)
	MODE = parameters.MODE
	load_weights = parameters.LOAD_WEIGHTS
	BATCH_SIZE = parameters.BATCH_SIZE
	EPOCHS = parameters.EPOCHS
	Z = parameters.Z
	D, G = parameters.D, parameters.G
	S = parameters.S
	data_path = parameters.data_path
	conv_kernel_init = parameters.conv_kernel_init
	dense_kernel_init = parameters.dense_kernel_init
	D_input_noise = parameters.D_input_noise
	D_dropout = parameters.D_dropout
	# create figure with two subplots
	fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(7, 9.6))
	fig = plt.gcf()
	fig.canvas.set_window_title('HISTORY')
	textstr = ('BATCH_SIZE =' + str(BATCH_SIZE) + 
				' | D/G = ' + str(D) + '/' + str(G)  +
				' | S = ' + str(S))
	fig.suptitle(textstr, fontsize=10)
	# subplot 1 
	ax0.set_ylabel('Loss')
	ax0.legend(loc='upper center', 
		bbox_to_anchor=(0.5, 1.15),
		fancybox=False, shadow=True, ncol=2)
	ax0.grid(True)
	# subplot 2
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Accuracy')
	ax1.grid(True)
	# update
	plt.ion()
	ax0.plot(y, '#50964B', label='Generator')
	ax0.plot(x, '#B1B1B1', label='Discriminator') 
	ax1.plot(Y, '#50964B' )
	ax1.plot(X, '#B1B1B1')
	# mode
	if MODE == 1:
		train(BATCH_SIZE, EPOCHS, D, G, S, data_path, parameters.G_optimizer, parameters.D_optimizer)
	elif MODE == 0:
		generate(BATCH_SIZE)
	else:
		print("No mode chosen in the *parameters.py* file!")

