import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plot
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

print('--------------loading data---------------')

(trainX, trainY), (testX, testY) = mnist.load_data()

print('--------------normalizing data---------------')

trainX = trainX.reshape(-1, 28, 28, 1) / 255
testX = testX.reshape(-1, 28, 28, 1) / 255
trainY = to_categorical(trainY)
testY = to_categorical(testY)

print('--------------defining model---------------')

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

print('--------------compiling model---------------')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(trainX, trainY, validation_split=0.2, epochs=10, batch_size=32, verbose=0)
score = model.evaluate(testX, testY, verbose=0)
print(f'Test Accuracy: {score[1]:.3f}')

plot.plot(history.history['accuracy'], label='Train Accuracy')
plot.plot(history.history['val_accuracy'], label='Validation Accuracy')
plot.legend()
plot.title('Accuracy vs. Epochs')
plot.xlabel('Epochs')
plot.ylabel('Accuracy')
plot.show()

plot.plot(history.history['loss'], label='Train Loss')
plot.plot(history.history['val_loss'], label='Validation Loss')
plot.legend()
plot.title('Loss vs. Epochs')
plot.xlabel('Epochs')
plot.ylabel('Loss')
plot.show()