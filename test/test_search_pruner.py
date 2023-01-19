from copy import deepcopy

import pytest

from irisml_tasks_automl import FlexibleBaseConfig, ConfigVarAccessor, TrainLog, SinglePeakPruner


class FakeConfig(FlexibleBaseConfig):
    def __init__(self, var_1, var_2):
        self.var_1 = var_1
        self.var_2 = var_2


class Var1Accessor(ConfigVarAccessor):

    def parse_value(self, config):
        return config.var_1

    def assign_val_to_config(self, config, val):
        result = deepcopy(config)
        result.var_1 = val
        return result


@pytest.mark.parametrize("history,remaining_candidates", [
    ([(2, 1), (3, 2)], [2, 3, 4, 5]),
    ([(1, 1), (3, 2)], [1, 2, 3, 4, 5]),
    ([(1, 3), (3, 2)], [1, 2, 3]),
    ([(1, 3), (2, 2)], [1, 2]),
    ([(1, 3), (5, 6)], [1, 2, 3, 4, 5]),
    ([(2, 3), (3, 6), (4, 5)], [2, 3, 4])
])
def test_single_peak_pruner(history, remaining_candidates):
    history = [TrainLog(FakeConfig(x[0], 1), {'automl_metric_val': x[1]}) for x in history]

    sp = SinglePeakPruner(Var1Accessor())
    result = sp.prune(FakeConfig(1, 1), [1, 2, 3, 4, 5], history)
    assert result == remaining_candidates
