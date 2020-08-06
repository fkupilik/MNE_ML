from sklearn import metrics


class LinearClassifier:
    """
    Class represents the linear classification model
    """

    def __init__(self, model):
        """
        Initializes the linear classification model
        :param model: scikit-learn classification model
        """
        self.model = model

    def fit(self, x_train, y_train, x_val, y_val):
        """
        Fits the model
        :param x_train: training samples
        :param y_train: training labels
        :param x_val: validation samples
        :param y_val: validation labels
        :return: fitted model
        """
        self.model.fit(x_train, y_train[:, 0])
        return self.evaluate(x_val, y_val)

    def evaluate(self, x_test, y_test):
        """
        Evaluates the fitted model
        :param x_test: test samples
        :param y_test: test labels
        :return: required metrics
        """
        predictions = []
        real_outputs = []
        for i in range(x_test.shape[0]):
            pattern = x_test[i, :].reshape(1, -1)
            prediction = self.model.predict(pattern)
            predictions.append(prediction[0])
            real_outputs.append(y_test[i, 0])
        auc = metrics.roc_auc_score(real_outputs, predictions)
        acc = metrics.accuracy_score(real_outputs, predictions)
        prec = metrics.precision_score(real_outputs, predictions)
        recall = metrics.recall_score(real_outputs, predictions)
        return [auc, acc, prec, recall]
