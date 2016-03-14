# -*- coding: utf-8 -*-
'''
Post-estimation reporting methods.
'''
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline


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

