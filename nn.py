from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import backend as K
import keras_metrics


def auc(y_true, y_pred):
    """
    Compute the Area under the ROC curve metric
    :param y_true: correct labels
    :param y_pred: predicted labels
    :return: AUC metric
    """
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


class NN:
    """
    Parent class for both types of neural networks
    """

    def compile(self):
        """
        Assign the loss function, optimizer and metrics to the model
        :return: model
        """
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=[auc, 'binary_accuracy', keras_metrics.precision(), keras_metrics.recall()])

    def fit(self, x_train, y_train, x_val, y_val):
        """
        Fits the model
        :param x_train: training samples
        :param y_train: training labels
        :param x_val: validation samples
        :param y_val: validation labels
        :return: trained model
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
        hist = self.model.fit(x_train, y_train, epochs=self.param.epochs, batch_size=16, shuffle=True,
                              callbacks=[early_stopping], verbose=self.param.verbose,
                              validation_data=(x_val, y_val))
        val_metrics = [hist.history['val_auc'][-1], hist.history['val_binary_accuracy'][-1],
                       hist.history['val_precision'][-1], hist.history['val_recall'][-1]]
        return val_metrics

    def evaluate(self, x_test, y_test):
        """
        Evaluates the fitted model
        :param x_test: test samples
        :param y_test: test labels
        :return: required metrics
        """
        metrics = self.model.evaluate(x_test, y_test, verbose=self.param.verbose)
        del metrics[0]
        return metrics
