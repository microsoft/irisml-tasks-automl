from abc import ABC, abstractmethod
from typing import List

from ..common.train_log import TrainLog
from ..common.base_config import ConfigVarAccessor


class CandidatePruner(ABC):
    """
    Pruner in config searching process. This class defines the logic pruning the candidates not worth trying based on history
    """

    def prune(self, base_config, candidates_in_order: List, history: List[TrainLog]):
        return [x for x in candidates_in_order if self.is_valuable(base_config, x, candidates_in_order, history)]

    @abstractmethod
    def is_valuable(self, base_config, candidate, candidates: List, history: List[TrainLog]):
        pass


class SinglePeakPruner(CandidatePruner):
    """
    This class prunes based on two assumptions:
    1. candidates are in monotonic order
    2. metric values for the candidates has a single peak, e.g. [can1:1, can2:2, can3:3, can4:2, can5:1] is valid, while [can1:1, can2:2, can3:1, can3:2] is not valid (two peaks)

    so, when we see history [can1:1, can2:2, can3:3, can4:2], there is no need to try can5, as can3 performs better than can4
    """

    def __init__(self, var_accessor: ConfigVarAccessor):
        self._var_accessor = var_accessor

    def is_valuable(self, base_config, candidate, candidates, history):
        candidate_config = self._var_accessor.assign_val_to_config(base_config, candidate)
        candidate_configs = [self._var_accessor.assign_val_to_config(base_config, x) for x in candidates]
        return self.worth_trying_one_side(candidate_config, candidate_configs, history) and self.worth_trying_one_side(candidate_config, reversed(candidate_configs), history)

    def worth_trying_one_side(self, candidate_config, candidate_configs, history):
        highest_val = None
        for x in candidate_configs:
            if x == candidate_config:
                return True

            metric_val = self._try_find_metric_val_for_config(x, history)
            if metric_val:
                if highest_val and metric_val < highest_val:
                    return False

                highest_val = metric_val

        raise RuntimeError('Candidate not in candidate list.')

    @staticmethod
    def _try_find_metric_val_for_config(config, history):
        for x in history:
            if x.config == config:
                return x.automl_metric_val

        return None
