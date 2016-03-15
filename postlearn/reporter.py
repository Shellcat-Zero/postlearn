# -*- coding: utf-8 -*-
'''
Post-estimation reporting methods.
'''
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt

def model_from_pipeline(pipe):
    if isinstance(pipe, Pipeline):
        return pipe[-1][1]
    else:
        return pipe

def extract_grid_scores(model):
    '''
    Extract grid scores from a model or pipeline.
    '''
    model = model_from_pipeline(model)
    return model.grid_scores_


def unpack_grid_scores(model=None):
    scores = extract_grid_scores(model)
    rows = []
    params = sorted(scores[0].parameters)

    for row in scores:
        mean = row.mean_validation_score
        std = row.cv_validation_scores.std()
        rows.append([mean, std] + [row.parameters[k] for k in params])
    return pd.DataFrame(rows, columns=['mean_', 'std_'] + params)

def plot_grid_scores(model, x, y, hue=None, row=None, col=None, col_wrap=None):
    scores = unpack_grid_scores(model)
    return sns.factorplot(x=x, y=y, hue=hue, row=row, col=col, data=scores,
                          col_wrap=col_wrap)


def plot_roc_curve(y_true, y_score, ax=None):
    '''
    '''
    ax = ax or plt.axes()
    score = metrics.roc_auc_score(y_true, y_score)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.annotate(score)


def plot_regularization_path(model):
    pass


class ClassificationResults:

    def __init__(self, model, X_train, y_train, X_test=None, y_test=None,
                 labels=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self._y_pred_train = None
        self._y_pred_test = None
        self._proba_train = None
        self._proba_test = None

        if labels is None:
            labels = sorted(np.unique(y_train))
        self.labels = labels

    @property
    def y_pred_train(self):
        if self._y_pred_train is None:
            self._y_pred_train = self.model.predict(self.X_train)
        return self._y_pred_train

    @property
    def y_pred_test(self):
        if self._y_pred_test is None:
            self._y_pred_test = self.model.predict(self.X_test)
        return self._y_pred_test

    @property
    def proba_train(self):
        if self._proba_train is None:
            self._proba_train = self.model.predict_proba(self.X_train)
        return self._proba_train

    @property
    def proba_test(self):
        if self._proba_test is None:
            self._proba_test = self.model.predict_proba(self.X_test)
        return self._proba_test

    def confusion_matrix(self, y=None, y_pred=None):
        if y is None and y_pred is None:
            y = self.y_train
            y_pred = self.y_pred_train
        df = (pd.DataFrame(metrics.confusion_matrix(y, y_pred),
                           index=self.labels, columns=self.labels)
                .rename_axis("actual")
                .rename_axis("predicted", axis=1))
        return df

