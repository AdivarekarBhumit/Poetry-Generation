import os
import numpy as np 
import pandas as pd  
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, RNN
from keras.utils import np_utils

##Loading Data
text = (open('./data/sonnets.txt').read())
text = text.lower()

##Create Character Mappings
characters = sorted(list(set(text)))

n_to_char = {n: char for n,char in enumerate(characters)}
char_to_n = {char: n for n,char in enumerate(characters)}

##Data Preprocessing
X = []
Y = []
length = len(text)
seq_len = 100

for i in range(0, length-seq_len, 1):
    sequence = text[i:i + seq_len]
    label = text[i + seq_len]
    X.append([char_to_n[char] for char in sequence])
    Y.append([char_to_n[label]])

X_modified = np.reshape(X, (len(X), seq_len, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)

## Our Model
model = Sequential()
model.add(LSTM(250, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(250, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(250))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_modified, Y_modified, epochs=100, batch_size=100)

with open('./models/poem_generation_model.json', 'w') as json_file:
    json_file.write(model.to_json())

model.save_weights('./models/poem_generation_model.h5')
print('Model Trained and Stored on Disk Successfully...')