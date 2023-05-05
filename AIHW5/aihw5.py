import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


data = pd.read_csv('C:/Users/jspac/OneDrive/Desktop/AIHW5/IMDB Dataset.csv')

print('preprocess data')
data['sentiment'] = data['sentiment'].replace({'positive': 1, 'negative': 0})
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['review'].values)
sequences = tokenizer.texts_to_sequences(data['review'].values)
paddedSeq = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')

print('build Long Short Term Memory model')
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=500))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

print('compile model')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print('train model')
history = model.fit(paddedSeq, data['sentiment'].values, epochs=25, batch_size=64, validation_split=0.2)

print('evaluate model')
testLoss, testAccuracy = model.evaluate(paddedSeq, data['sentiment'].values, verbose=2)
print("Test Loss: ", testLoss)
print("Test Accuracy: ", testAccuracy)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Epoch vs Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Epouch vs. Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()