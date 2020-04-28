import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import Constant
import matplotlib.pyplot as plt
import numpy as np
import random


# Courtesy https://www.tensorflow.org/tutorials/keras/classification#feed_the_model
def plot_image(idx, predictions_array, true_label, img):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    predictions_array, true_label_array, img = predictions_array, true_label[idx], img[idx]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    true_label = np.argmax(true_label_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


# Courtesy https://www.tensorflow.org/tutorials/keras/classification#feed_the_model
def plot_value_array(idx, predictions_array, true_label):
    predictions_array, true_label_array = predictions_array, true_label[idx]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    true_label = np.argmax(true_label_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Courtesy https://www.tensorflow.org/tutorials/images/classification#data_augmentation
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def plot_random_prediction(predictions, test_labels, test_images):
    i = random.randint(0, len(predictions))
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, np.squeeze(test_images))
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()


print(tf.__version__)
tf.config.list_physical_devices('GPU')

# An AlexNet variation
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
model = Sequential()
model.add(BatchNormalization(input_shape=(28, 28, 1)))
# Block 1
model.add(Conv2D(16, (5, 5), activation='relu', padding='same', bias_initializer=Constant(0.01),
          kernel_initializer='random_uniform'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

# Block 2
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', bias_initializer=Constant(0.01),
          kernel_initializer='random_uniform'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', bias_initializer=Constant(0.01),
          kernel_initializer='random_uniform'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

# Block 3
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=Constant(0.01),
          kernel_initializer='random_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', bias_initializer=Constant(0.01),
          kernel_initializer='random_uniform'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))

# Fully connected
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(10))

# Optimizer
model.compile(optimizer=RMSprop(learning_rate=0.0005),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
model.summary()

# Data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
N = len(test_images) - 1000
validation_images = test_images[N:]
validation_labels = test_labels[N:]
test_images = test_images[:N]
test_labels = test_labels[:N]

# Data transformation
# Courtesy https://medium.com/@lukaszlipinski/fashion-mnist-with-keras-in-5-minuts-20ab9eb7b905
train_images = tf.expand_dims(train_images, -1)
test_images = test_images.astype('float32') / 255.
test_images = tf.expand_dims(test_images, -1)
validation_images = validation_images.astype('float32') / 255.
validation_images = tf.expand_dims(validation_images, -1)
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)
validation_labels = tf.keras.utils.to_categorical(validation_labels, 10)

# Data augmentation
# datagen = ImageDataGenerator(rescale=1./255)
datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=15,
                             width_shift_range=.1,
                             height_shift_range=.1,
                             horizontal_flip=True)

# Training params
epochs = 10
batch_size = 32

# Training with data augmentation
history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=batch_size),
                              steps_per_epoch=len(train_images) / 32,
                              epochs=epochs,
                              validation_data=(validation_images, validation_labels))

# Training
# model.fit(train_images,
#           train_labels,
#           epochs=epochs,
#           batch_size=batch_size)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('\nTest accuracy:', test_acc)

# Plotting training and testing over time
plot_history(history)

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
# plot_random_prediction(predictions, test_labels, test_images)
