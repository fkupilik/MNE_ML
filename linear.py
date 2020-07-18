from sklearn import metrics


class LinearClassifier:

    def __init__(self, model):
        self.model = model

    def fit(self, x_train, y_train, x_val, y_val):
        self.model.fit(x_train, y_train[:, 0])
        return self.evaluate(x_val, y_val)

    def evaluate(self, x_test, y_test):
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
