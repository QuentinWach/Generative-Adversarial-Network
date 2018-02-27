# preprocess.py

from Keras import MNIST
import


#============================================================================
# MNIST Dataset

# format MNIST dataset from Keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, :, :, None]
X_test = X_test[:, :, :, None]



#============================================================================
# Custom Tree Dataset

# format the curstom scraped dataset

# split dataset into training and testing with input 
(X_train, y_train), (X_test, y_test) = 