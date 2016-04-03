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

