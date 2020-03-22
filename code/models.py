import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from process import generate_data
from sklearn.model_selection import train_test_split

class Model():
	def __init__(self, split=0.2):
		pass

	def train_model(self, epochs):
		pass

	def generate_buckets(self):
		self.buckets = []
		#plays with less than two minutes left in the half
		temp = self.test[self.test['UTM'].isin(['1'])]
		self.buckets.append(('under two minutes', temp))
		#plays with different field positions
		temp = self.test[(self.test['field_pos'] >= 20) & (self.test['field_pos'] < 30)]
		self.buckets.append(('20-30', temp))
		temp = self.test[(self.test['field_pos'] >= 30) & (self.test['field_pos'] < 40)]
		self.buckets.append(('30-40', temp))
		temp = self.test[(self.test['field_pos'] >= 40) & (self.test['field_pos'] < 50)]
		self.buckets.append(('40-50', temp))
		temp = self.test[(self.test['field_pos'] >= 50) & (self.test['field_pos'] < 60)]
		self.buckets.append(('50-60', temp))
		temp = self.test[(self.test['field_pos'] >= 60) & (self.test['field_pos'] < 70)]
		self.buckets.append(('60-70', temp))
		temp = self.test[(self.test['field_pos'] >= 70) & (self.test['field_pos'] < 80)]
		self.buckets.append(('70-80', temp))
		temp = self.test[(self.test['field_pos'] >= 80) & (self.test['field_pos'] < 90)]
		self.buckets.append(('80-90', temp))
		temp = self.test[(self.test['field_pos'] >= 90) & (self.test['field_pos'] < 100)]
		self.buckets.append(('90-100', temp))

		#plays with a large score differential
		temp = self.test[(self.test['score_differential'] > 21)]
		self.buckets.append(('score diff > 21', temp))

		#plays in the last quarter of the game with a reasonable score differential
		temp = self.test[(self.test['score_differential'] < 14) & (self.test['time_remaining'] <= 900) & self.test['is_half_two'] == 1]
		self.buckets.append(('4th quarter low score diff', temp))

	def evaluate(self):
		#determine accuracy of model
		print("*****************")
		print(self.__class__.__name__)
		print("*****************")
		for b in self.buckets:
			x = b[1].drop(columns=['y'])
			y = np.stack(b[1]['y'])
			predicted = [0] * y.shape[1]
			actual = [0] * y.shape[1]
			for row in self.model.predict(x):
				predicted = [sum(i) for i in zip(predicted, row)]

			for row in y:
				actual = [sum(i) for i in zip(actual, row)]

			mse = 0
			for j in range(0,len(predicted)):
				mse += (predicted[j] - actual[j]) ** 2
			print(b[0], mse / y.shape[1])

#######################################
# OutcomeModel is a neural network that, given the state of the game at the 
# beginning of a drive, will predict the outcome of the drive.
#######################################
class OutcomeModel(Model):
	def __init__(self, split=0.2):
		self.data = generate_data()
		self.train, self.test =  train_test_split(self.data, test_size=split, random_state=0)
		self.model = Sequential()		

	def train_model(self, epochs):
		x_train = self.train.drop(columns=['y'])
		y_train = np.stack(self.train['y'])
		self.model.add(Dense(70, activation='relu', input_dim=x_train.shape[1]))
		self.model.add(Dropout(0.1))
		self.model.add(Dense(70, activation='relu'))
		self.model.add(Dense(len(y_train[0]), activation='softmax'))
		self.model.compile(optimizer='adam',
	          loss='categorical_crossentropy',
	          metrics=['accuracy'])
		self.model.fit(x_train, y_train, epochs=epochs)

#######################################
# YardDistributionModel is a neural network that, given the state of the game at a fourth
#down, will predict the yards gained by the offense at that point.
#######################################
class YardDistributionModel(Model):
	def __init__(self, split=0.2):
		self.data = generate_data(True)
		self.train, self.test =  train_test_split(self.data, test_size=split, random_state=0)
		self.train = self.train.dropna()
		self.test = self.test.dropna()
		self.model = Sequential()
	
	def train_model(self, epochs):
		x_train = self.train.drop(columns=['y'])
		y_train = np.stack(self.train['y'])
		self.model.add(Dense(70, activation='relu', input_dim=x_train.shape[1]))
		self.model.add(Dropout(0.1))
		self.model.add(Dense(70, activation='relu'))
		self.model.add(Dense(len(y_train[0]), activation='softmax'))
		self.model.compile(optimizer='adam',
	          loss='categorical_crossentropy',
	          metrics=['accuracy'])
		self.model.fit(x_train, y_train, epochs=epochs)
		
