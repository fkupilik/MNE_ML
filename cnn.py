from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Activation, Dropout, Input, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras_metrics
from keras.callbacks import EarlyStopping
#import output.plots as display
import tensorflow as tf
from keras import backend as K

from nn import NN


class CNN(NN):

    def __init__(self, channels, time_samples, param):
        self.model = Sequential()
        self.model.add(Conv2D(6, (3, 3), activation='elu', input_shape=(channels, time_samples, 1)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(AveragePooling2D(pool_size=(1, 8)))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='elu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2, activation='softmax'))
        self.param = param
        self.compile()



