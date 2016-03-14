# -*- coding: utf-8 -*-
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScalar
from sklearn.pipeline import make_pipeline

import postlearn as pl

@pytest.fixture
def X():
    np.random.seed(42)
    return np.random.randn(10, 2)

def y():
    np.random.seed(32)
    return np.random.uniform(0, 1, size=10)

@pytest.fixture
def model():
    return LinearRegression()

@pytest.fixture
def pipeline(model):
    return make_pipeline([StandardScalar(), model])


@pytest.mark.parametrize("arg", [model, pipeline])
def test_model_from_pipeline(arg, model):
    result = pl.model_from_pipeline(model)
    assert result == model
