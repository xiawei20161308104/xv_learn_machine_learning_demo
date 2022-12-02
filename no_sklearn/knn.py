'''
Author: xv rg16xw@163.com
Date: 2022-12-02 09:46:29
LastEditors: xiawei
LastEditTime: 2022-12-02 14:00:28
FilePath: \xv_learn_machine_learning_demo\no_sklearn\knn.py
Description: knn realization
'''
import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:
    def __init__(self, k):
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0],   \
            "x must be equal y"
        assert self.k <= X_train.shape[0],\
            'x must be more than k'
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        assert self._X_train is not None and self._y_train is not None,\
            "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1],\
            "predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        # dis = (x_train-x)**2
        # num_dis = np.sum((x_train-x)**2)
        # sqrt_sum=sqrt(np.sum((x_train-x)**2))

        distances = [sqrt(np.sum((x_train-x)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        top_k_y = [self._y_train[i] for i in nearest[:self.k]]
        score = Counter(top_k_y)
        return score.most_common(1)[0][0]

    def __repr__(self) -> str:
        return 'KNN(k=%d)' % self.k
