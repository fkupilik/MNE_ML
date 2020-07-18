from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout

from nn import NN


class RNN(NN):

    def __init__(self, channels, time_samples, param):
        dropout = .25
        self.model = Sequential()
        self.model.add(LSTM(input_shape=(channels, time_samples), units=100, return_sequences=True, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout))

        self.model.add(LSTM(units=50, return_sequences=False, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout))

        self.model.add(Dense(units=2, activation='softmax'))
        self.param = param
        self.compile()




