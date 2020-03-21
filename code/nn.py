import pandas as pd
import numpy as np
import os
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier



no_score_only = False
split = 0.2
data = pd.read_csv("../data/drive_data.csv")
data = data.drop(columns=['game_id', 'drive_number'])
if no_score_only:
	data = data[data['result'].isin(['fourth_down'])]

encoder = LabelEncoder()
y = data['yds_gained'] if no_score_only else data['result']
encoder.fit(y)
encoded_y = encoder.transform(y)
labels = np_utils.to_categorical(encoded_y)
new_y = pd.Series(list(labels))
data['y'] = new_y
data = data.drop(columns=['result', 'yds_gained'])

train, test =  train_test_split(data, test_size=split, random_state=0)
x_train = train.drop(columns=['y'])
y_train = np.stack(train['y'])
x_test = test.drop(columns=['y'])
y_test = np.stack(test['y'])


os.environ["CUDA_VISIBLE_DEVICES"]="-1"

buckets = []

#plays with less than two minutes left in the half
temp = data[data['UTM'].isin(['1'])]
buckets.append(('under two minutes', temp))

#plays with different field positions
temp = data[(data['field_pos'] >= 20) & (data['field_pos'] < 30)]
buckets.append(('20-30', temp))
temp = data[(data['field_pos'] >= 30) & (data['field_pos'] < 40)]
buckets.append(('30-40', temp))
temp = data[(data['field_pos'] >= 40) & (data['field_pos'] < 50)]
buckets.append(('40-50', temp))
temp = data[(data['field_pos'] >= 50) & (data['field_pos'] < 60)]
buckets.append(('50-60', temp))
temp = data[(data['field_pos'] >= 60) & (data['field_pos'] < 70)]
buckets.append(('60-70', temp))
temp = data[(data['field_pos'] >= 70) & (data['field_pos'] < 80)]
buckets.append(('70-80', temp))
temp = data[(data['field_pos'] >= 80) & (data['field_pos'] < 90)]
buckets.append(('80-90', temp))
temp = data[(data['field_pos'] >= 90) & (data['field_pos'] < 100)]
buckets.append(('90-100', temp))

#plays with a large score differential
temp = data[(data['score_differential'] > 21)]
buckets.append(('score diff > 21', temp))

#plays in the last quarter of the game with a reasonable score differential
temp = data[(data['score_differential'] < 14) & (data['time_remaining'] <= 900) & data['is_half_two'] == 1]
buckets.append(('4th quarter low score diff', temp))

model = Sequential()
model.add(Dense(70, activation='relu', input_dim=x_train.shape[1]))
model.add(Dropout(0.1))
model.add(Dense(70, activation='relu'))
model.add(Dense(len(y_train[0]), activation='softmax'))
model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])
model.fit(x_train, y_train, epochs=500)


for b in buckets:
	x = b[1].drop(columns=['y'])
	y = np.stack(b[1]['y'])
	predicted = [0] * y.shape[1]
	actual = [0] * y.shape[1]
	for row in model.predict(x):
		predicted = [sum(i) for i in zip(predicted, row)]

	for row in y:
		actual = [sum(i) for i in zip(actual, row)]

	mse = 0
	for j in range(0,len(predicted)):
		mse += (predicted[j] - actual[j]) ** 2
	print(b[0], mse / y.shape[1])
	print(predicted)
	print(actual)



# nbrs = KNeighborsClassifier(n_neighbors=50)
# fit = nbrs.fit(x_train, y_train)
# predicted = [0] * test_labels.shape[1]
# actual = [0] * test_labels.shape[1]
# for row in fit.predict_proba(x_test):
# 	predicted = [sum(x) for x in zip(predicted, row)]

# for row in test_labels:
# 	actual = [sum(x) for x in zip(actual, row)]

# mse = 0
# for j in range(0,len(predicted)):
# 	mse += (predicted[j] - actual[j]) **2
# mse /= test_labels.shape[1]
# print(predicted)
# print(actual)
# print(mse)

# dtc = DecisionTreeClassifier(random_state=0)
# model = dtc.fit(x_train, y_train)
# predicted = [0] * y_test.shape[1]
# actual = [0] * y_test.shape[1]
# print(model.predict_proba(x_test))
# for row in model.predict_proba(x_test):
# 	predicted = [sum(x) for x in zip(predicted, row)]

# for row in y_test:
# 	actual = [sum(x) for x in zip(actual, row)]

# mse = 0
# for j in range(0,len(predicted)):
# 	mse += (predicted[j] - actual[j]) ** 2
# mse /= y_test.shape[1]
# print(predicted)



