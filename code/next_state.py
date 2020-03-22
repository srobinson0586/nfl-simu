import os
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


def train_outcome(x_train, y_train):
	os.environ["CUDA_VISIBLE_DEVICES"]="-1"
	model = Sequential()
	model.add(Dense(70, activation='relu', input_dim=x_train.shape[1]))
	model.add(Dropout(0.1))
	model.add(Dense(70, activation='relu'))
	model.add(Dense(len(y_train[0]), activation='softmax'))
	model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
	model.fit(x_train, y_train, epochs=500)

	return model