# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and trainin

#print(train_labels[:10])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Show sample data image
# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Data preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

# Layer 1: This is our input layer and it will conist of 784 neurons. 
# We use the flatten layer with an input shape of (28,28) 
# to denote that our input should come in in that shape. 
# The flatten means that our layer will reshape the shape (28,28) array 
# into a vector of 784 neurons so that each pixel will be associated with one neuron.

# Layer 2: This is our first and only hidden layer. 
# The dense denotes that this layer will be fully connected: 
# each neuron from the previous layer connects to each neuron of this layer.
# It has 128 neurons and uses ReLu activation function

# Layer 3: This is our output later and is also a dense layer. 
# It has 10 neurons that we will look at to determine our models
# output. Each neuron represnts the probabillity of a given image 
# being one of the 10 different classes. 
# The activation function softmax is used on this layer 
# to calculate a probabillity distribution for each class. 
# This means the value of any neuron in this layer will be between 0 and 1, 
# where 1 represents a high probabillity of the image being that class.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # input layer (1)
    keras.layers.Dense(128, activation='relu'), # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])

# define the loss function, optimizer and metrics we would like to track.
model.compile(optimizer='adam', 
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

# we pass the data, labels and epochs and watch the magic!
model.fit(train_images, train_labels, epochs=10)

# "verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar."
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 1)

# print('Test accuracy: ', test_acc)

predictions = model.predict(test_images)
print("prediction: ", predictions[0])
print("best score: ", np.argmax(predictions[0]), " with label:", test_labels[0])
