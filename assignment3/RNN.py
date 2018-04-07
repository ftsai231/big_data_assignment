#!/usr/bin/env python

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import scipy.io
import csv
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from pickle import load
import xlsxwriter
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Input, Flatten
from keras.layers.convolutional import Convolution2D
import keras

# load .mat file into dictionary x
x = scipy.io.loadmat('/home/tsai/assignment3/train_sequence.mat')
training_data = x['R']

y = scipy.io.loadmat('/home/tsai/assignment3/test_sequence.mat')
predict_data = y['test_R']

training_answer = pd.read_csv('train_result.csv',sep=' ',header=None)
training_answer = np.asarray(training_answer)

model = Sequential()
model.add(Embedding(14, 1))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#model.load_weights("weights.best.hdf5")  
filepath ="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
history = model.fit(training_data, training_answer,epochs=150,verbose=2,validation_split=0.1,callbacks=[checkpoint],shuffle=True)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

predictions = model.predict_classes(predict_data)
workbook = xlsxwriter.Workbook('arrays.xlsx')
worksheet = workbook.add_worksheet()
row=0
col=0
for row, data in enumerate(predictions):
    worksheet.write(row, col, data)
workbook.close()