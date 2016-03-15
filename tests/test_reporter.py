# -*- coding: utf-8 -*-
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import postlearn as pl

@pytest.fixture
def X():
    np.random.seed(42)
    return np.random.randn(10, 2)

@pytest.fixture
def y():
    np.random.seed(32)
    return np.random.uniform(0, 1, size=10)

@pytest.fixture
def y_discrete():
    np.random.seed(32)
    return np.random.randint(0, 2, size=10)


@pytest.fixture
def regression_model():
    return LinearRegression()

@pytest.fixture
def classification_model():
    return LogisticRegression()


@pytest.fixture
def regression_pipeline(regression_model):
    return make_pipeline([StandardScaler(), regression_model])


@pytest.fixture
def fit_regression_model(regression_model, X, y):
    return regression_model.fit(X, y)


@pytest.fixture
def fit_classification_model(classification_model, X, y_discrete):
    return classification_model.fit(X, y_discrete)


@pytest.fixture
def result(fit_classification_model, X, y_discrete):
    return pl.ClassificationResults(fit_classification_model, X, y_discrete)


@pytest.mark.parametrize("arg", [regression_model, regression_pipeline])
def test_model_from_pipeline(arg, regression_model):
    result = pl.model_from_pipeline(regression_model)
    assert result == regression_model


def test_confusion_matrix(result):
    cm = result.confusion_matrix()
    assert cm.index.name == 'actual'
    assert cm.columns.name == 'predicted'


def test_plot_roc_curve(result):
    result.plot_roc_curve(y_true=result.y_train)

