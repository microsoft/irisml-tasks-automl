from abc import ABC, abstractmethod


class CostEstimator(ABC):
    """
    Base class for cost estimator, which estimates the cost of training a dataset with a given config
    """
    def __init__(self):
        pass

    @abstractmethod
    def estimate(self, train_config, dataset):
        pass
