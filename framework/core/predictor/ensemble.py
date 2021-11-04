"""Base Estimator"""
# pylint: disable=bare-except
import math
import pickle


class EnsemblePredictor():
    def __init__(self, config, logger=None, reporter=None, name=None):
        self._reporter = reporter
        name = name if isinstance(name, str) else self.__class__.__name__
        self._name = name
        # reserved attributes
        self.net = None
        self.num_class = None

