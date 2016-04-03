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
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt

try:
    from ipywidgets import interact
    has_widgets = True
except ImportError:
    has_widgets = False


def model_from_pipeline(pipe):
    '''
    Extract the model from the last stage of a pipeline.

    Parameters
    ----------
    pipe : Pipeline or Estimator

    Returns
    -------

    model: Estimator
    '''
    if isinstance(pipe, Pipeline):
        return pipe[-1][1]
    else:
        return pipe


def extract_grid_scores(model):
    '''
    Extract grid scores from a model or pipeline.

    Parameters
    ----------
    model : Estimator or Pipeline
        must end in sklearn.grid_search.GridSearchCV

    Returns
    -------
    scores : list

    See Also
    --------
    unpack_grid_scores
    '''
    model = model_from_pipeline(model)
    return model.grid_scores_


def unpack_grid_scores(model=None):
    '''
    Unpack mean grid scores into a DataFrame

    Parameters
    ----------
    model : Estimator or Pipeline
        must end in sklearn.grid_search.GridSearchCV

    Returns
    -------
    scores : DataFrame

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn import datasets
    >>> from sklearn.grid_search import GridSearchCV
    >>> from sklearn.preprocessing import StandardScaler
    >>> X, y =datasets.make_classification()
    >>> model = GridSearchCV(RandomForestClassifier(),
    ...                      param_grid={
    ...                          'n_estimators': [10, 20, 30],
    ...                          'max_features': [.1, .5, 1]
    ...                      })
    >>> model.fit(X, y)
    >>> unpack_grid_scores(model)
       mean_      std_  max_features  n_estimators
    0   0.88  0.062416           0.1            10
    1   0.88  0.046536           0.1            20
    2   0.85  0.095309           0.1            30
    3   0.88  0.062686           0.5            10
    4   0.91  0.072044           0.5            20
    5   0.90  0.073366           0.5            30
    6   0.78  0.032929           1.0            10
    7   0.86  0.048224           1.0            20
    8   0.85  0.072174           1.0            30
    '''
    scores = extract_grid_scores(model)
    rows = []
    params = sorted(scores[0].parameters)
    for row in scores:
        mean = row.mean_validation_score
        std = row.cv_validation_scores.std()
        rows.append([mean, std] + [row.parameters[k] for k in params])
    return pd.DataFrame(rows, columns=['mean_', 'std_'] + params)


def plot_grid_scores(model, x, y, hue=None, row=None, col=None, col_wrap=None,
                     **kwargs):
    '''
    Wrapper around seaborn.factorplot.

    Parameters
    ----------
    model : Pipeline or Estimator
    x, hue, row, col : str
        parameters grid searched over
    y : str
        the target of interest, probably `'mean_'`

    Returns
    -------
    g : seaborn.FacetGrid
    '''
    scores = unpack_grid_scores(model)
    return sns.factorplot(x=x, y=y, hue=hue, row=row, col=col, data=scores,
                          col_wrap=col_wrap, **kwargs)


def plot_roc_curve(y_true, y_score, ax=None):
    '''
    Plot the Receiving Operator Characteristic curved, including the
    Area under the Curve (AUC).

    Parameters
    ----------
    y_true : array
    y_score : array
    ax : matplotlib.axes, defaults to new axes

    Returns
    -------
    ax : matplotlib.axes
    '''
    ax = ax or plt.axes()
    auc = metrics.roc_auc_score(y_true, y_score)
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.annotate('AUC: {:.2f}'.format(auc), (.8, .2))
    ax.plot([0, 1], [0, 1], linestyle='--', color='k')
    return ax


def plot_regularization_path(model):
    '''
    Plot the regularization path of coefficients from e.g. a Lasso
    '''
    raise ValueError


def plot_learning_curve(estimator, X, y, train_sizes=np.linspace(.1, 1.0, 5),
                        cv=None, n_jobs=1, ax=None):
    '''

    '''
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return ax


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
    '''Sort a Series by magnitude, ignoring direction (sign).'''
    return s[np.abs(s).argsort()]


def plot_feature_importance(model, labels, n=10, orient='h'):
    '''
    Bar plot of feature importance.

    Parameters
    ----------
    model : Pipeline or Estimator
    labels : list-like
    n : int
        number of features to include
    orient : {'h', 'v'}
        horizontal or vertical barplot

    Returns
    -------
    ax : matplotlib.axes

    Notes
    -----
    Works with Regression, coefs_, or ensembes with feature_importances_

    '''
    model = model_from_pipeline(model)
    if orient.lower().startswith('h'):
        kind = 'barh'
    elif orient.lower().startswith('v'):
        kind = 'bar'
    else:
        raise ValueError("`orient` should be 'v' or 'h', got %s instead" %
                         orient)
    features = _get_feature_importance(model)
    if labels:
        features = features.reshape(len(labels), -1)
    features = (pd.DataFrame(features, index=labels)
                  .squeeze()
                  .pipe(_magsort)
                  .tail(n)
                  .plot(kind=kind))
    return features


def confusion_matrix(y_true=None, y_pred=None, labels=None):
    '''
    Dataframe of confusion matrix. Rows are actual, and columns are predicted.

    Parameters
    ----------
    y_true : array
    y_pred : array
    labels : list-like

    Returns
    -------
    confusion_matrix : DataFrame
    '''
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
            sig = inspect.signature(func)
            keys = {k for k in sig.parameters if k != 'self'}
            for kw in keys:
                # TODO: Broken
                if kwargs.get(kw) is None:
                    kwargs[kw] = getattr(self, attrs.get(kw, kw),
                                         sig.parameters[kw].default)
            return func(self, **kwargs)
        return wrapper
    return deco


class GridSearchMixin:

    def plot_grid_scores(self, x, hue=None, row=None, col=None, col_wrap=None,
                         **kwargs):
        def none_if_none(x):
            return None if x == 'None' else x

        if has_widgets:
            choices = ['None'] + list(unpack_grid_scores(self.model)
                                      .columns.drop(['mean_', 'std_']))

            @interact(x=choices, hue=choices, row=choices, col=choices)
            def wrapper(x=x, hue=None, row=None, col=None):
                return plot_grid_scores(self.model,
                                        none_if_none(x),
                                        'mean_',
                                        hue=none_if_none(hue),
                                        row=none_if_none(row),
                                        col=none_if_none(col),
                                        col_wrap=none_if_none(col_wrap),
                                        **kwargs)
            return wrapper
        else:
            return plot_grid_scores(self.model, x, 'mean_', hue=hue, row=row,
                                    col=col, col_wrap=col_wrap, **kwargs)


class ClassificationResults(GridSearchMixin):
    '''
    A convinience class, wrapping all the reporting methods and
    caching intermediate calculations.
    '''

    def __init__(self, model, X_train, y_train, X_test=None, y_test=None,
                 labels=None):
        '''
        Parameters
        ----------
        model : Pipeline or Estimator
        X_train : np.array
        y_train : np.array
        X_test : np.array
        y_test : np.array
        labels : list of str
        '''
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
        'Predicted values for the training set'
        if self._y_pred_train is None:
            self._y_pred_train = self.model.predict(self.X_train)
        return self._y_pred_train

    @property
    def y_pred_test(self):
        'Predicted values for the test set'
        if self._y_pred_test is None:
            self._y_pred_test = self.model.predict(self.X_test)
        return self._y_pred_test

    @property
    def proba_train(self):
        'Predicted probabilities for the training set'
        if self._proba_train is None:
            self._proba_train = self.model.predict_proba(self.X_train)
        return self._proba_train

    @property
    def proba_test(self):
        'Predicted probabilities for the test set'
        if self._proba_test is None:
            self._proba_test = self.model.predict_proba(self.X_test)
        return self._proba_test

    @property
    def y_score_train(self):
        'Predicted positive score (column 1) for the training set'
        if self._y_score_train is None:
            self._y_score_train = self.proba_train[:, 1]
        return self._y_score_train

    @property
    def y_score_test(self):
        'Predicted positive score (column 1) for the test set'
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
        Plot the ROC.
        '''
        return plot_roc_curve(y_true=y_true, y_score=y_score, ax=ax)

    def make_report(self):
        pass

    @default_args(X='X_train', y='y_train')
    def plot_learning_curve(self, *, X=None, y=None,
                            train_sizes=None, cv=None,
                            n_jobs=1,
                            ax=None):
        return plot_learning_curve(self.model, X=X, y=y, cv=cv,
                                   train_sizes=train_sizes,
                                   n_jobs=n_jobs,
                                   ax=ax)

