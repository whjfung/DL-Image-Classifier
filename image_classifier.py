# Description: This programe is inspired by Youtube Channel Computer Science https://www.youtube.com/watch?v=iGWbqhdjf2s with some imprpvements
#              A Machine Learning program that uses TensorFlow & Convolutional Neural Networks (CNN) to classify images
#              Trained with University of Toronto's CIFAR-100 dataset.

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Load Data
from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# Check data types of variables
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

# Get shape of arrays
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# Look at the first image as an array
index = 0
x_train[index]

# Show image as img
img = plt.imshow(x_train[index])

# Get image label
print('The image label is: ', y_train[index])

# Image classification defined by CIFAR
classification = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

# Print Image Class
print('The image class is : ', classification[y_train[index][0]])

# Convert the labels into a set of 10 numbers to input into the neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Print the new labels
print(y_train_one_hot)
print(y_test_one_hot)

# Print the new label of the image/ picture above
print('The one hot label is: ', y_train_one_hot[index])

# Normalize pixels to values between 0 and 1
x_train = x_train/255
x_test = x_test/255

# Create the models architecture
model = Sequential()

# Add the first layer
model.add(Conv2D(32, (5,5), activation='relu', input_shape = (32,32,3)))

# Add a pooling layer
model.add(MaxPooling2D(pool_size = (2,2)))

# Add another convolution layer
model.add(MaxPooling2D(pool_size = (2,2)))

# Add a flattening layer
model.add(Flatten())

# Add a layer with 2000 neurons
model.add(Dense(2000, activation = 'relu'))

# Add a dropout layer
model.add(Dropout(0.5))

# Add a layer with 1000 neurons
model.add(Dense(1000, activation = 'relu'))

# Add a dropout layer
model.add(Dropout(0.5))

# Add a layer with 500 neurons
model.add(Dense(500, activation = 'relu'))

# Add a layer with 250 neurons
model.add(Dense(250, activation = 'relu'))

# Add a layer with 100 neurons
model.add(Dense(100, activation = 'softmax'))

# Compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# Train the model
hist = model.fit(x_train, y_train_one_hot, batch_size=1, epochs=10, validation_split=0.2)

# Evaluate the model using the test data set
model.evaluate(x_test, y_test_one_hot)[1]

# Visualize the model's accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()

# Visualize the model loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()

# Resize the image
new_image = plt.imread("sample_data/whale.jpg")
from skimage.transform import resize
resized_image = resize(new_image, (32, 32, 3))

# Show the images
img = plt.imshow(new_image)
img = plt.imshow(resized_image)

# Get the model's prediction
predictions = model.predict(np.array([resized_image]))
# Show prediction
print(predictions)

# Sort the predictions in accending order
list_index = np.arange(100).tolist()
x = predictions

for i in range(100):
  for j in range(100):
    if x[0][list_index[i]] > x[0][list_index[j]]:
      temp = list_index[i]
      list_index[i] = list_index[j]
      list_index[j] = temp

# Show the sorted labels in order
print(list_index)

# Print the first 5 predictions
print('Classification :')
for i in range(5):
  print(classification[list_index[i]], ':', round(predictions[0][list_index[i]] * 100, 2), '%')