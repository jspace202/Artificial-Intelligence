import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout
import tensorflow as tf

print('--------------loading data---------------')

(trainX, trainY), (testX, testY) = cifar10.load_data()

print('--------------normalizing data...---------------')

trainX = trainX / 255
testX = testX / 255
trainY = to_categorical(trainY)
testY = to_categorical(testY)

print('--------------defining model---------------')

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print('--------------compiling model...---------------')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('--------------training model---------------')

model.fit(trainX, trainY, validation_split=0.2, epochs=10, batch_size=32, verbose=1)

print('--------------evaluting model---------------')

score = model.evaluate(testX, testY, verbose=0)
print(f'Test accuracy: {score[1]:.3f}')

print('--------------loading ResNet 50 model---------------')

resnet_model = ResNet50(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
resnet_model.layers.pop()

trainXresize = tf.image.resize(trainX, (128, 128))
testXresize = tf.image.resize(testX, (128, 128))

model = Sequential()
model.add(resnet_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

print('--------------compiling ResNet 50 model---------------')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('--------------training ResNet 50 model---------------')

model.fit(trainXresize, trainY, validation_split=0.2, epochs=10, batch_size=32, verbose=1)

print('--------------evaluating ResNet 50 model---------------')

score = model.evaluate(testXresize, testY, verbose=0)
print(f'Test accuracy: {score[1]:.3f}') 