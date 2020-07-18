from sklearn.svm import SVC

import data_loading
import linear
import cnn
import rnn

import numpy as np
from sklearn import preprocessing

from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from param import Param


def windowed_means(out_features, param):
    sampling_fq = param.t_max * 1000 + 1
    temp_wnd = np.linspace(param.min_latency, param.max_latency, param.steps + 1)
    intervals = np.zeros((param.steps, 2))
    for i in range(0, temp_wnd.shape[0] - 1):
        intervals[i, 0] = temp_wnd[i]
        intervals[i, 1] = temp_wnd[i + 1]
    intervals = intervals - param.t_min
    output_features = []
    for i in range(out_features.shape[0]):
        feature = []
        for j in range(out_features.shape[1]):
            time_course = out_features[i][j]
            for k in range(intervals.shape[0]):
                borders = intervals[k] * sampling_fq
                feature.append(np.average(time_course[int(borders[0] - 1):int(borders[1] - 1)]))
        output_features.append(feature)
    out = preprocessing.scale(np.array(output_features), axis=1)
    return out

#choose classifier
type = 'cnn' # choose from 'svm', 'lda', 'cnn', 'rnn'

param = Param()

X, Y = data_loading.read_data(param)

if type == 'cnn':
    X = np.expand_dims(X, 3)
elif type == 'lda' or type == 'svm':
    X = windowed_means(X, param)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=param.test_part,
                                                    random_state=0, shuffle=True)

val = round(param.validation_part * x_train.shape[0])

shuffle_split = ShuffleSplit(n_splits=param.cross_val_iter, test_size=val, random_state=0)
val_results = []
test_results = []

for train, validation in shuffle_split.split(x_train):

    if type == 'cnn':
        model = cnn.CNN(x_train.shape[1], x_train.shape[2], param)
    elif type == 'rnn':
        model = rnn.RNN(x_train.shape[1], x_train.shape[2], param)
    elif type == 'lda':
        model = linear.LinearClassifier(LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto'))
    else:
        model = linear.LinearClassifier(SVC(cache_size=500))

    validation_metrics = model.fit(x_train[train], y_train[train], x_train[validation], y_train[validation])
    val_results.append(validation_metrics)

    test_metrics = model.evaluate(x_test, y_test)
    test_results.append(test_metrics)

print("\nClassifier: ", type)

avg_val_results = np.round(np.mean(val_results, axis=0) * 100, 2)
avg_val_results_std = np.round(np.std(val_results, axis=0) * 100, 2)

print("Averaged validation results with averaged std in brackets:")
print("AUC: ", avg_val_results[0], "(", avg_val_results_std[0], ")")
print("accuracy: ", avg_val_results[1], "(", avg_val_results_std[1], ")")
print("precision: ", avg_val_results[2], "(", avg_val_results_std[2], ")")
print("recall: ", avg_val_results[3], "(", avg_val_results_std[3], ")")

print("\n##############################\n")

avg_test_results = np.round(np.mean(test_results, axis=0) * 100, 2)
avg_test_results_std = np.round(np.std(test_results, axis=0) * 100, 2)

print("Averaged test results with averaged std in brackets: ")
print("AUC: ", avg_test_results[0], "(", avg_test_results_std[0], ")")
print("accuracy: ", avg_test_results[1], "(", avg_test_results_std[1], ")")
print("precision: ", avg_test_results[2], "(", avg_test_results_std[2], ")")
print("recall: ", avg_test_results[3], "(", avg_test_results_std[3], ")")

