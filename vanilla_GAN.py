import numpy as np; np.random.seed(42)
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, AveragePooling2D, LeakyReLU
from keras.layers import BatchNormalization, UpSampling2D, Conv2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img

# Generator (TODO: DCGAN)
def createGenerator():
	g = Sequential(name="Generator")
	g.add(Dense(49152, input_shape=(100,)))
	g.add(LeakyReLU(alpha=0.3))
	g.add(Reshape((8, 8, 768)))
	g.add(UpSampling2D(size=(2, 2)))
	g.add(Conv2D(192, (5, 5), padding='same'))
	g.add(LeakyReLU(alpha=0.3))
	g.add(Dropout(0.5, seed=12))
	g.add(UpSampling2D(size=(2, 2)))
	g.add(Conv2D(48, (5, 5), padding='same'))
	g.add(LeakyReLU(alpha=0.3))
	g.add(UpSampling2D(size=(2, 2)))
	g.add(Conv2D(12, (5, 5), padding='same'))
	g.add(Dropout(0.5, seed=7))
	g.add(LeakyReLU(alpha=0.3))
	g.add(UpSampling2D(size=(2, 2)))
	g.add(Conv2D(3, (5, 5), padding='same', activation="tanh"))
	
	#print("GENERATOR:")
	#g.summary()
	return g


# Discriminator
def createDiscriminator():
	d = Sequential(name="Discriminator")
	#d.add(BatchNormalization(axis=-1, input_shape=(128, 128, 3)))
	d.add(Conv2D(64, (5, 5), padding='same', input_shape=(128, 128, 3)))
	d.add(LeakyReLU(alpha=0.3))
	d.add(AveragePooling2D(pool_size=(2, 2)))
	d.add(Conv2D(128, (5, 5)))
	d.add(LeakyReLU(alpha=0.3))
	d.add(AveragePooling2D(pool_size=(2, 2)))
	d.add(Flatten())
	d.add(Dense(1024))
	d.add(LeakyReLU(alpha=0.3))
	d.add(Dense(1, activation="sigmoid"))

	#print("DISCRIMINATOR:")
	#d.summary() # for debugging
	return d


# Train the GAN
def train(BATCH_SIZE, EPOCHS):

	# Initialize the GAN
	gan = Sequential()
	g = createGenerator()
	d = createDiscriminator()
	d.trainable = False
	gan.add(g)
	gan.add(d)
	#print("GENERATIVE ADVERSARIAL NETWORK:")
	#gan.summary()

	# compile the GAN
	g.compile(loss='binary_crossentropy', optimizer="SGD")
	g.load_weights('history/generator.h5')

	gan.compile(loss='binary_crossentropy', optimizer="SGD")
	d.trainable = True
	d.compile(loss='binary_crossentropy', optimizer="SGD")
	d.load_weights('history/discriminator.h5')


	# augmentation configuration for training
	real_batch = ImageDataGenerator(
		rescale=1. / 255, 
		shear_range=0.2, 
		zoom_range=0.2, 
		horizontal_flip=True,
		data_format="channels_last")

	# enable iterating through the directory
	real_batch = real_batch.flow_from_directory(
		"data", 
		target_size=(128, 128),
		color_mode="rgb",	
		batch_size=BATCH_SIZE,	
		class_mode=None,
		shuffle=True,
		seed=42)

	real_batch = next(real_batch)
	real_batch = np.array(real_batch)


	# The actual algorithm as described in the original paper (with k=1):
	# ===================================================================
	for epoch in range(EPOCHS):
		print("=================================================================")
		print("Epoch "+ str(epoch + 1))
		print("_________________________________________________________________")
		
		# 1. Sample minibatch of m noise samples from noise prior
		noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100)) 

		# 2. Sample minibatch of m examples from data generating distribution
		fake_batch = np.array(g.predict(noise, verbose=0))

		# 3. Update the discriminator D by ascending its stochastic greadient
		data = np.concatenate((real_batch, fake_batch))
		label = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
		d_loss = d.train_on_batch(x=data, y=label)
		print("Discriminator loss : %f" % (d_loss))	

		# 4. Sample minibatch of m noise samples (redundant for k=1)
		noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

		# 5. Update the generator G by descenging its stochastic gradient
		d.trainable = False
		g_loss = gan.train_on_batch(x=noise, y=np.array([1] * BATCH_SIZE))
		print("Generator loss : %f" % (g_loss))
		d.trainable = True

		# (save weights)
		g.save_weights('history/generator.h5', True)
		d.save_weights('history/discriminator.h5', True)

		# show and save loss and accuracy

		# select the image array of the complete array of generated images 
		pic = (np.vsplit(fake_batch, BATCH_SIZE))[0] 	# select one image
														# of the set of gen. images
		# reformat image array
		pic = pic*255
		pic = np.squeeze(pic, axis=0)
		pic = pic.astype(np.uint8)

		# save image array as jpg in diretory
		Image.fromarray(pic, "RGB").save("gen/Img_"+str(epoch+101)+".jpg")
		print("...sample image saved as "+"Img_"+str(epoch+101)+".jpg")


# Generate and save images to directory
def generate(BATCH_SIZE):

	# compile generator
	g = createGenerator()
	g.compile(loss='binary_crossentropy', optimizer="SGD")

	# load weights of the generator
	g.load_weights('history/generator.h5')

	# create noise for generator
	noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))

	# create array of a set of images
	generated_images = g.predict(noise, verbose=1)

	# save each image individually to the cdr
	for sample in range(BATCH_SIZE):

		# select the image array of the complete array of generated images 
		pic = (np.vsplit(generated_images, BATCH_SIZE))[sample]

		# reformat image array
		pic = pic*255
		pic = np.squeeze(pic, axis=0)
		pic = pic.astype(np.uint8)

		# save image array as jpg in diretory
		Image.fromarray(pic, "RGB").save("Img_"+str(sample+1)+".jpg")


# Run
if __name__ == "__main__":
	BATCH_SIZE = 64
	EPOCHS = 1000
	train(BATCH_SIZE, EPOCHS)
	#generate(5)




