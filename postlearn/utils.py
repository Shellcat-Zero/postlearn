import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


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


def discrete_cmap(N, base_cmap=None):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def rediscretize_cmap(N, base_cmap=None):
    base = plt.cm.get_cmap(base_cmap)
    step = base.N // N

    if hasattr(base, 'colors'):
        colors = base.colors[0:base.N, step]
    else:
        colors = base(np.arange(0, base.N, step))
    name = base.name + str(N)
    return mpl.colors.ListedColormap(colors, name, N)
