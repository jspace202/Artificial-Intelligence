import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plot

print("loading dataset...")
wine = load_wine()
X = wine.data
Y = wine.target

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)

print("defining model...")
model = Sequential()
model.add(Dense(64, input_shape=(trainX.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

print("compiling model...")
model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("training model...")
history = model.fit(trainX, trainY, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

plot.plot(history.history['loss'], label='Training Loss', color='r')
plot.plot(history.history['val_loss'], label='Validation Loss', color='g')
plot.xlabel('Iterations')
plot.ylabel('Loss')
plot.legend()
plot.show()
plot.plot(history.history['accuracy'], label='Training Accuracy', color='r')
plot.plot(history.history['val_accuracy'], label='Validation Accuracy', color='g')
plot.xlabel('Iterations')
plot.ylabel('Accuracy')
plot.legend()
plot.show()

testValLoss, testAccuracy = model.evaluate(testX, testY, verbose=0)
print(f'Test Accuracy: {testAccuracy:.3f}')