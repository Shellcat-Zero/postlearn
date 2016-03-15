# -*- coding: utf-8 -*-
'''
Post-estimation reporting methods.
'''
import inspect
from functools import wraps

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
    auc = metrics.roc_auc_score(y_true, y_score)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.annotate('AUC: {:.2f}'.format(auc), (.8, .2))
    ax.plot([0, 1], [0, 1], linestyle='--', color='k')
    return ax

def plot_regularization_path(model):
    raise ValueError

def _get_feature_importance(model):
    '''
    LinearModels: `.coef_`
    ensemble: `feature_importances_`
    '''
    order = ['coef_', 'feature_importances_']
    for attr in order:
        if hasattr(model, attr):
            return getattr(model, attr)
    else:
        raise ValueError("The model does not have any of {}".format(order))


def _magsort(s):
    return s[np.abs(s).argsort()]

def plot_feature_importance(model, labels, n=10, orient='h'):
    if orient.lower().startswith('h'):
        kind = 'barh'
    elif orient.lower().startswith('v'):
        kind = 'bar'
    else:
        raise ValueError("`orient` should be 'v' or 'h', got %s instead" %
                         orient)
    features = (pd.DataFrame(_get_feature_importance(model),
                             index=labels)
                  .squeeze()
                  .pipe(_magsort)
                  .tail(n)
                  .plot(kind=kind))
    return features


def confusion_matrix(y_true=None, y_pred=None, labels=None):
    df = (pd.DataFrame(metrics.confusion_matrix(y_true, y_pred),
                       index=labels, columns=labels)
            .rename_axis("actual")
            .rename_axis("predicted", axis=1))
    return df


def default_args(**attrs):
    '''
    Pull the defaults for a method from `self`.

    Parameters
    ----------
    attrs : dict
        mapping parameter name to attribute name
        Attributes with the same name need not be included.

    Returns
    -------
    deco: new function, injecting the `attrs` into `kwargs`

    Notes
    -----
    Only usable with keyword-only arguments.

    Examples
    --------

    @default_args({'y': 'y_train'})
    def printer(self, *, y=None, y_pred=None):
        print('y: ', y)
        print('y_pred: ', y_pred)
    '''
    def deco(func):
        @wraps(func)
        def wrapper(self, **kwargs):
            sig = {k for k in inspect.signature(func).parameters
                   if k != 'self'}
            keys = kwargs.keys() | sig
            for kw in keys:
                if kwargs.get(kw) is None:
                    kwargs[kw] = getattr(self, attrs.get(kw, kw), None)
            return func(self, **kwargs)
        return wrapper
    return deco


class ClassificationResults:
    '''
    A convinience class.
    '''

    def __init__(self, model, X_train, y_train, X_test=None, y_test=None,
                 labels=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self._y_score_train = None
        self._y_score_test = None
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

    @property
    def y_score_train(self):
        if self._y_score_train is None:
            self._y_score_train = self.proba_train[:, 1]
        return self._y_score_train

    @property
    def y_score_test(self):
        if self._y_score_test is None:
            self._y_score_test = self.proba_test[:, 1]
        return self._y_score_test

    @default_args(y_true='y_train', y_pred='y_pred_train')
    def confusion_matrix(self, y_true=None, y_pred=None, labels=None):
        return confusion_matrix(y_true=y_true,
                                y_pred=y_pred,
                                labels=labels)

    @default_args(y_true='y_train', y_score='y_score_train')
    def plot_roc_curve(self, *, ax=None, y_true=None, y_score=None):
        '''
        '''
        return plot_roc_curve(y_true=y_true, y_score=y_score, ax=ax)

