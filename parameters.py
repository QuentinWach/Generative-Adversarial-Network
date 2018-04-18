from keras.optimizers import Adam, SGD
from keras import initializers
from keras import layers
from keras.layers import BatchNormalization

#=============================================================================
""" INITILIZATION"""

MODE = 1							# 1: Train the GAN, 0: Generate samples
EPOCHS = 5000						# number of training iterations
LOAD_WEIGHTS = 0					# 1: True, 0: False
BATCH_SIZE = 32						# number of samples per iteration
SEED = 41							# seed for pseudo random number generator
Z = 100								# noise input dimension
D, G = 1, 1							# step balance between D and G per epoch
S = 0.0								# strength of label smoothing


#conv_kernel_init = 'glorot_uniform'		
#conv_kernel_init = 'random_uniform'
conv_kernel_init = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=SEED)

#dense_kernel_init = 'glorot_uniform'
#dense_kernel_init = 'random_uniform'
dense_kernel_init = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=SEED)


#=============================================================================
""" REGULARIZATION """

D_input_noise = 0.5					# additive zero-centered Gaussian noise
D_dropout = 0.5						# drop probability of multiplicative 
									# 1-centered Gaussian noise

#=============================================================================
""" OPTIMIZATION """

"""
G_optimizer = SGD()
D_optimizer = SGD()
"""

# Generator (Adam)
G_LR = 0.0003
G_BETA_1 = 0.5
G_BETA_2 = 0.999
G_EPSILON = None
G_DECAY = 0.0
G_AMSGRAD = False

G_optimizer = Adam(lr=G_LR, beta_1=G_BETA_1, beta_2=G_BETA_2, 
	epsilon=G_EPSILON, decay=G_DECAY, amsgrad=G_AMSGRAD)

"""
# Discriminator (Statistical Gradient Descent)

D_LR = 0.12
D_MOM = 0.0
D_DECAY = 0.3
D_NEST = False

D_optimizer = SGD(lr=D_LR, momentum=D_MOM, decay=D_DECAY, nesterov=D_NEST)
"""

# Discriminator (Adam)
D_LR = 0.0002
D_BETA_1 = 0.999
D_BETA_2 = 0.999
D_EPSILON = None
D_DECAY = 0.0
D_AMSGRAD = False

D_optimizer = Adam(lr=D_LR, beta_1=D_BETA_1, beta_2=D_BETA_2, 
	epsilon=D_EPSILON, decay=D_DECAY, amsgrad=D_AMSGRAD)



#=============================================================================
""" DATASET """

#data_path = "forests"
data_path = "celebA"
#data_path = "cats"
#data_path = "dogs"


