from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout

from nn import NN


class RNN(NN):
    """
    Class for the LSTM neural network initialization
    """

    def __init__(self, channels, time_samples, param):
        """
        Initializes the LSTM neural network
        :param channels: number of the channels
        :param time_samples: number of the time samples
        :param param: configuration object
        """
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




