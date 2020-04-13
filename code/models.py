import numpy as np
import math
import os
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from process import generate_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


class Model():
    def __init__(self, split=0.2):
        self.data, self.classes = generate_data(self.__class__.__name__)

    def train_model(self, epochs, split=0.2):
        self.train, self.test = train_test_split(
            self.data, test_size=split, random_state=0)
        self.model = Sequential()
        x_train = self.train.drop(columns=['y'])
        y_train = np.stack(self.train['y'])
        self.model.add(Dense(70, activation='relu',
                             input_dim=x_train.shape[1]))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(80, activation='relu'))
        self.model.add(Dense(len(y_train[0]), activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=1024, verbose=0)

    def generate_buckets(self):
        self.buckets = []
        # plays with less than two minutes left in the half
        self.buckets.append(('all', self.test))
        # plays with different field positions
        temp = self.test[(self.test['field_pos'] >= 20) &
                         (self.test['field_pos'] < 30)]
        self.buckets.append(('20-30', temp))
        temp = self.test[(self.test['field_pos'] >= 30) &
                         (self.test['field_pos'] < 40)]
        self.buckets.append(('30-40', temp))
        temp = self.test[(self.test['field_pos'] >= 40) &
                         (self.test['field_pos'] < 50)]
        self.buckets.append(('40-50', temp))
        temp = self.test[(self.test['field_pos'] >= 50) &
                         (self.test['field_pos'] < 60)]
        self.buckets.append(('50-60', temp))
        temp = self.test[(self.test['field_pos'] >= 60) &
                         (self.test['field_pos'] < 70)]
        self.buckets.append(('60-70', temp))
        temp = self.test[(self.test['field_pos'] >= 70) &
                         (self.test['field_pos'] < 80)]
        self.buckets.append(('70-80', temp))
        temp = self.test[(self.test['field_pos'] >= 80) &
                         (self.test['field_pos'] < 90)]
        self.buckets.append(('80-90', temp))
        temp = self.test[(self.test['field_pos'] >= 90) &
                         (self.test['field_pos'] < 100)]
        self.buckets.append(('90-100', temp))

        # plays with a large score differential
        temp = self.test[(self.test['score_differential'] > 21)]
        self.buckets.append(('score diff > 21', temp))

        # plays in the last quarter of the game with a reasonable score differential
        temp = self.test[(self.test['score_differential'] < 14) & (
            self.test['time_remaining'] <= 900) & self.test['is_half_two'] == 1]
        self.buckets.append(('4th quarter low score diff', temp))

    def evaluate(self):
        # determine accuracy of model
        self.generate_buckets()
        print()
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
            for j in range(0, len(predicted)):
                mse += (predicted[j] - actual[j]) ** 2
            print(b[0], mse / y.shape[1])

    def predict(self,state):
        return self.model.predict(np.array([state,]))[0]
#######################################
# OutcomeModel is a neural network that, given the state of the game at the
# beginning of a drive, will predict the outcome of the drive.
#######################################


class OutcomeModel(Model):
    def train_model(self, epochs, split=0.2):
        self.train, self.test = train_test_split(
            self.data, test_size=split, random_state=0)
        self.model = Sequential()
        x_train = self.train.drop(columns=['y'])
        y_train = np.stack(self.train['y'])
        self.model.add(Dense(80, activation='relu',
                             input_dim=x_train.shape[1]))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(70, activation='relu'))
        self.model.add(Dense(len(y_train[0]), activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=1024, verbose=0)

#######################################
# YardDistributionModel is a neural network that, given the state of the game at a fourth
# down, will predict the yards gained by the offense at that point.
#######################################


class YardDistributionModel(Model):
    def train_model(self, epochs, split=0.2):
        self.train, self.test = train_test_split(
            self.data, test_size=split, random_state=0)
        self.model = Sequential()
        x_train = self.train.drop(columns=['y'])
        y_train = np.stack(self.train['y'])
        self.model.add(Dense(70, activation='relu',
                             input_dim=x_train.shape[1]))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(70, activation='relu'))
        self.model.add(Dense(len(y_train[0]), activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=1024, verbose=0)


class TimeRunoffModel(Model):
    pass


class TurnoverFieldPosModel(Model):
    def generate_buckets(self):
        self.buckets = [('all',self.test)]


class FieldGoalModel(Model):
    def train_model(self):
        attempts = {}
        made = {}
        for i in range(15, 70, 5):
            attempts[i] = 0
            made[i] = 0

        for y in self.data['y']:
            y = int(y)
            y = min(y, 65)
            y = max(y, -65)
            attempts[5 * math.floor(abs(y) / 5)] += 1
            if y > 0:
                made[5 * math.floor(abs(y) / 5)] += 1

        self.model = {}
        for key in attempts:
            self.model[key] = made[key] / attempts[key]
        

    def evaluate(self):
        print()
        print("*****************")
        print(self.__class__.__name__)
        print("*****************")
        for key in self.model:
            if key == 65:
                print("Category: 65+, Percentage: %.2f%%" %
                      (self.model[key] * 100))
            else:
                print("Category: %d-%d, Percentage: %.2f%%" %
                      (key, key + 5, self.model[key] * 100))
    def predict(self, yds):
        return self.model[min(65, 5 * math.floor(yds / 5))]
        


class SmallYardDistributionModel(Model):
    def __init__(self):
        self.data, self.classes = generate_data(self.__class__.__name__)
        self.classes = [i for i in range(0,10)]

    def train_model(self):
        groups = {}
        yards = {}
        self.model = {}
        for y in self.data['y']:
            y = int(y)
            group = int(y / 10)
            if group not in groups:
                groups[group] = 0
                yards[group] = [0] * 10

            groups[group] += 1
            yards[group][y % 10] += 1
        for key in yards:
            self.model[key] = [x / groups[key] for x in yards[key]]

    def evaluate(self):
        print()
        print("*****************")
        print(self.__class__.__name__)
        print("*****************")
        for key in self.model:
            print("Catergory: %d-%d yds:" % (key * 10, key * 10 + 10))
            print(["{0:0.2f}".format(i) for i in self.model[key]])
    
    def predict(self, yds):
        return self.model[yds]


class PuntModel(Model):

    def train_model(self, neighbors=50, split=0.2):
        self.train, self.test = train_test_split(self.data, test_size=split, random_state=0)
        x_train = self.train.drop(columns=['y'])
        y_train = self.train['y']
        model = KNeighborsClassifier(n_neighbors=neighbors)
        model.fit(x_train, y_train)
        self.classes = model.classes_
        self.model = [0] * 101
        for yds in range(0,101):
            self.model[yds] = model.predict_proba(np.array([[yds]]))[0]
            
    def generate_buckets(self):
        self.buckets = [('all', self.test)]
    
    def evaluate(self):
        print()
        print("*****************")
        print(self.__class__.__name__)
        print("*****************")
        self.generate_buckets()
        x_test = self.test.drop(columns=['y'])
        y_test = self.test['y']

        encoder = LabelEncoder()
        encoder.fit(self.data['y'])
        encoded_y = encoder.transform(y_test)           
        labels = np_utils.to_categorical(encoded_y)
        y_test = np.array(labels)
        
            
           
        predictions = np.stack(self.model.predict_proba(x_test))
        predicted = [0] * predictions.shape[1]
        actual = [0] * predictions.shape[1]
        for row in predictions:
            predicted = [sum(i) for i in zip(predicted, row)]
        for row in y_test:
            actual = [sum(i) for i in zip(actual, row)]

        mse = 0
        for j in range(0, len(predicted)):
            mse += (predicted[j] - actual[j]) ** 2
        print('all', mse / predictions.shape[1])
    
    def predict(self, yds):
        return self.model[yds]

class FourthDownModel (Model):

    def train_model(self):
        attempts = {}
        made = {}
        for i in range(0, 35, 5):
            attempts[i] = 0
            made[i] = 0

        for y in self.data['y']:
            y = int(y)
            y = min(y, 30)
            y = max(y, -30)
            attempts[5 * math.floor(abs(y) / 5)] += 1
            if y > 0:
                made[5 * math.floor(abs(y) / 5)] += 1

        self.model = {}
        for key in attempts:
            self.model[key] = made[key] / attempts[key]

    def evaluate(self):
        print()
        print("*****************")
        print(self.__class__.__name__)
        print("*****************")
        for key in self.model:
            if key == 30:
                print("Category: 30+, Percentage: %.2f%%" %
                      (self.model[key] * 100))
            else:
                print("Category: %d-%d, Percentage: %.2f%%" %
                      (key, key + 5, self.model[key] * 100))

    def predict(self, yds):
        return self.model[min(30, 5 * math.floor(yds / 5))]

        

def train_models(epochs):
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    print("Training Outcome Model...")
    om = OutcomeModel()
    om.train_model(epochs)

    print("Training Yard Distribution Model...")
    yd = YardDistributionModel()
    yd.train_model(epochs)

    print("Training Field Goal Model...")
    fg = FieldGoalModel()
    fg.train_model()

    print("Training Small Yard Model...")
    sm = SmallYardDistributionModel() 
    sm.train_model()

    print("Training Punt Model...")
    pm = PuntModel()
    pm.train_model(50)

    print("Training Time Runoff Model...")
    tr = TimeRunoffModel()
    tr.train_model(epochs)

    print("Training Turnover Model...")
    to = TurnoverFieldPosModel()
    to.train_model(epochs)

    print("Training Fourth Down Model...")
    fd = FourthDownModel()
    fd.train_model()

    models = [om, yd, fg, sm, pm, tr, to, fd]
    return models
        
