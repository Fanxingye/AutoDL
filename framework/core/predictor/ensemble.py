"""Base Estimator"""
# pylint: disable=bare-except
import math
import pickle


class EnsemblePredictor():
    """This is the base estimator for gluoncv.auto.Estimators.

    Parameters
    ----------
    config : dict
        Config in nested dict.
    logger : logging.Logger
        Optional logger for this estimator, can be `None` when default setting is used.
    reporter : callable
        The reporter for metric checkpointing.
    name : str
        Optional name for the estimator.

    Attributes
    ----------
    _logger : logging.Logger
        The customized/default logger for this estimator.
    _logdir : str
        The temporary dir for logs.
    _cfg : autocfg.dataclass
        The configurations.

    """
    def __init__(self, config, logger=None, reporter=None, name=None):
        self._reporter = reporter
        name = name if isinstance(name, str) else self.__class__.__name__
        self._name = name
        # reserved attributes
        self.net = None
        self.num_class = None

    def fit(self, train_data, val_data=None, train_size=0.9, random_state=None,
            resume=False, time_limit=None):
        results = self._fit(train_data, val_data, time_limit=time_limit) 
        return results

    def evaluate(self, val_data):
        """Evaluate estimator on validation data.

        Parameters
        ----------
        val_data : pd.DataFrame or iterator
            The validation data.

        """
        return self._evaluate(val_data)

    def predict(self, x, **kwargs):
        """Predict using this estimator.

        Parameters
        ----------
        x : str, pd.DataFrame or ndarray
            The input, can be str(filepath), pd.DataFrame with 'image' column, or raw ndarray input.

        """
        return self._predict(x, **kwargs)

    def predict_feature(self, x):
        """Predict intermediate features using this estimator.

        Parameters
        ----------
        x : str, pd.DataFrame or ndarray
            The input, can be str(filepath), pd.DataFrame with 'image' column, or raw ndarray input.

        """
        return self._predict_feature(x)

    def _reload_best(self, return_value):
        """Applying the best checkpoint before return"""
        cp = return_value.get('checkpoint', '')
        if not cp:
            return return_value
        self._logger.info('Applying the state from the best checkpoint...')
        try:
            tmp = self.load(cp)
            self.__dict__.update(tmp.__dict__)
        except:
            self._logger.warning(
                'Unable to resume the state from the best checkpoint, using the latest state.')
        return return_value

    def _predict(self, x, **kwargs):
        raise NotImplementedError

    def _predict_feature(self, x, **kwargs):
        raise NotImplementedError

    def _fit(self, train_data, val_data, time_limit=math.inf):
        raise NotImplementedError

    def _resume_fit(self, train_data, val_data, time_limit=math.inf):
        raise NotImplementedError

    def _evaluate(self, val_data):
        raise NotImplementedError

    def _init_network(self):
        raise NotImplementedError

    def _init_trainer(self):
        raise NotImplementedError

    def save(self, filename):
        """Save the state of this estimator to disk.

        Parameters
        ----------
        filename : str
            The file name for storing the full state.
        """
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid)
        self._logger.debug('Pickled to %s', filename)

    @classmethod
    def load(cls, filename, ctx='auto'):
        """Load the state from disk copy.

        Parameters
        ----------
        filename : str
            The file name to load from.
        ctx: str, default is 'auto'
            The context for reloaded model.
            'auto': use previously saved context type if still available, fallback
            to cpu if no gpu detected.
            Use `cpu` if no GPU available.
            'cpu': use cpu for inference regardless.
            'gpu': use as many gpus available as possible.
            [0, 2, 4, ...]: if a list or tuple of integers are provided, the context
            will be [gpu(0), gpu(2), gpu(4)...]
        """
        with open(filename, 'rb') as fid:
            obj = pickle.load(fid)
            return obj