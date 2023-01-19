import pytest
from copy import deepcopy
from unittest import mock
from irisml_tasks_automl import SinglePeakPruner, SingleVarSearchController, GridSearchController, SearchDimension, StageWiseSearchController,\
    AlterDecorator, TrainLog, FlexibleBaseConfig, ConfigVarAccessor


class FakeConfig(FlexibleBaseConfig):
    def __init__(self, var_1, var_2):
        self.var_1 = var_1
        self.var_2 = var_2

    def print(self):
        print(self.var_1, self.var_2)


class Var1Accessor(ConfigVarAccessor):

    def parse_value(self, config):
        return config.var_1

    def assign_val_to_config(self, config, val):
        result = deepcopy(config)
        result.var_1 = val
        return result


class Var2Accessor(ConfigVarAccessor):

    def parse_value(self, config):
        return config.var_2

    def assign_val_to_config(self, config, val):
        result = deepcopy(config)
        result.var_2 = val
        return result


ONE_DIM_HISTORY = [
    [(2, 1), (3, 2)],
    [(1, 1), (3, 2)],
    [(1, 3), (3, 2)],
    [(1, 3), (2, 2)],
    [(1, 3), (5, 6)],
    [(2, 3), (3, 6), (4, 5)],
    []]


@pytest.mark.parametrize("history,expected_configs", [
    (ONE_DIM_HISTORY[0], [1, 4, 5]),
    (ONE_DIM_HISTORY[1], [2, 4, 5]),
    (ONE_DIM_HISTORY[2], [2, 4, 5]),
    (ONE_DIM_HISTORY[3], [3, 4, 5]),
    (ONE_DIM_HISTORY[4], [2, 3, 4]),
    (ONE_DIM_HISTORY[5], [1, 5]),
    (ONE_DIM_HISTORY[6], [1, 2, 3, 4]),
])
def test_discrete_var_search_controller(history, expected_configs):
    history = [TrainLog(FakeConfig(x[0], 1), {'automl_metric_val': x[1]}) for x in history]
    expected_configs = [FakeConfig(x, 1) for x in expected_configs]
    ce = mock.MagicMock()
    ce.estimate.return_value = 0

    controller = SingleVarSearchController(FakeConfig(1, 1), ce, mock.MagicMock(), [1, 2, 3, 4, 5], Var1Accessor())
    configs = controller.generate_training_configs(10000, history, 4)
    assert configs == expected_configs


@pytest.mark.parametrize("history,expected_configs", [
    (ONE_DIM_HISTORY[0], [5, 4, 1]),
    (ONE_DIM_HISTORY[1], [2, 5, 4]),
    (ONE_DIM_HISTORY[2], [2, 5, 4]),
    (ONE_DIM_HISTORY[3], [3, 5, 4]),
    (ONE_DIM_HISTORY[4], [3, 2, 4]),
    (ONE_DIM_HISTORY[5], [5, 1]),
    (ONE_DIM_HISTORY[6], [3, 2, 5, 4]),
])
def test_discrete_var_search_controller_order_given(history, expected_configs):
    history = [TrainLog(FakeConfig(x[0], 1), {'automl_metric_val': x[1]}) for x in history]
    expected_configs = [FakeConfig(x, 1) for x in expected_configs]
    ce = mock.MagicMock()
    ce.estimate.return_value = 0

    controller = SingleVarSearchController(FakeConfig(1, 1), ce, mock.MagicMock(), [3, 2, 5, 4, 1], Var1Accessor(), candidates_order=[1, 2, 3, 4, 5])
    configs = controller.generate_training_configs(10000, history, 4)
    assert configs == expected_configs


@pytest.mark.parametrize("history,n_trials,expected_configs", [
    (ONE_DIM_HISTORY[0], 4, [4, 5]),
    (ONE_DIM_HISTORY[1], 4, [2, 4, 5]),
    (ONE_DIM_HISTORY[1], 1, [2]),
    (ONE_DIM_HISTORY[2], 4, [2]),
    (ONE_DIM_HISTORY[3], 4, []),
    (ONE_DIM_HISTORY[4], 4, [2, 3, 4]),
    (ONE_DIM_HISTORY[5], 4, []),
    (ONE_DIM_HISTORY[6], 4, [1, 2, 3, 4]),
    (ONE_DIM_HISTORY[6], 0, []),
])
def test_discrete_var_search_controller_with_pruner(history, n_trials, expected_configs):
    history = [TrainLog(FakeConfig(x[0], 1), {'automl_metric_val': x[1]}) for x in history]
    expected_configs = [FakeConfig(x, 1) for x in expected_configs]
    ce = mock.MagicMock()
    ce.estimate.return_value = 0

    controller = SingleVarSearchController(FakeConfig(1, 1), ce, mock.MagicMock(), [1, 2, 3, 4, 5], Var1Accessor(), SinglePeakPruner(Var1Accessor()))
    configs = controller.generate_training_configs(10000, history, n_trials)

    assert configs == expected_configs


TWO_DIM_HISTORY = [
    [(2, 1, 1), (3, 2, 2)],
    [(1, 1, 1), (1, 3, 2)],
    [(1, 1, 3), (1, 2, 2)],
    [(1, 1, 3), (2, 1, 2), (1, 2, 2)],
    [(3, 4, 3), (4, 3, 3), (4, 4, 6)],
    [(2, 1, 3), (3, 1, 6), (4, 2, 5)],
    [],
    [(1, 1, 3), (2, 1, 3.5), (3, 1, 2), (4, 1, 1)],
    [(1, 1, 3), (2, 1, 3.5), (3, 1, 2), (2, 2, 3)]]


@pytest.mark.parametrize("history,n_trials,expected_configs", [
    (TWO_DIM_HISTORY[0], 2, [(1, 1), (1, 2)]),
    (TWO_DIM_HISTORY[1], 1, [(1, 2)]),
    (TWO_DIM_HISTORY[2], 3, [(1, 3), (1, 4), (2, 1)]),
    (TWO_DIM_HISTORY[3], 5, [(1, 3), (1, 4), (2, 2), (2, 3), (2, 4)]),
    (TWO_DIM_HISTORY[4], 4, [(1, 1), (1, 2), (1, 3), (1, 4)]),
    (TWO_DIM_HISTORY[5], 3, [(1, 1), (1, 2), (1, 3)]),
    (TWO_DIM_HISTORY[6], 3, [(1, 1), (1, 2), (1, 3)]),
])
def test_grid_search_controller(history, n_trials, expected_configs):
    history = [TrainLog(FakeConfig(x[0], x[1]), {'automl_metric_val': x[2]}) for x in history]
    expected_configs = [FakeConfig(x[0], x[1]) for x in expected_configs]
    ce = mock.MagicMock()
    ce.estimate.return_value = 0

    gs = GridSearchController(FakeConfig(1, 1), ce, None, [SearchDimension([1, 2, 3, 4], Var1Accessor()), SearchDimension([1, 2, 3, 4], Var2Accessor())])
    configs = gs.generate_training_configs(10000, history, n_trials)

    assert configs == expected_configs


@pytest.mark.parametrize("history,n_trials,expected_configs", [
    (TWO_DIM_HISTORY[0], 2, [(1, 1), (1, 2)]),
    (TWO_DIM_HISTORY[1], 1, [(1, 2)]),
    (TWO_DIM_HISTORY[2], 3, [(2, 1), (2, 2), (2, 3)]),
    (TWO_DIM_HISTORY[3], 5, [(2, 2), (2, 3), (2, 4), (3, 1), (3, 2)]),
    (TWO_DIM_HISTORY[4], 8, [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4)]),
    (TWO_DIM_HISTORY[5], 3, [(1, 1), (1, 2), (1, 3)]),
    (TWO_DIM_HISTORY[6], 3, [(1, 1), (1, 2), (1, 3)]),
])
def test_grid_search_controller_with_one_pruner(history, n_trials, expected_configs):
    history = [TrainLog(FakeConfig(x[0], x[1]), {'automl_metric_val': x[2]}) for x in history]
    expected_configs = [FakeConfig(x[0], x[1]) for x in expected_configs]
    ce = mock.MagicMock()
    ce.estimate.return_value = 0

    gs = GridSearchController(FakeConfig(1, 1), ce, None, [SearchDimension([1, 2, 3, 4], Var1Accessor()),
                                                           SearchDimension([1, 2, 3, 4], Var2Accessor(), SinglePeakPruner(Var2Accessor()))])
    configs = gs.generate_training_configs(10000, history, n_trials)

    assert configs == expected_configs


@pytest.mark.parametrize("history,n_trials,expected_configs", [
    (TWO_DIM_HISTORY[0], 2, [(1, 1), (1, 2)]),
    (TWO_DIM_HISTORY[1], 1, [(1, 2)]),
    (TWO_DIM_HISTORY[2], 3, [(2, 1), (2, 2), (2, 3)]),
    (TWO_DIM_HISTORY[3], 5, [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3)]),
    (TWO_DIM_HISTORY[4], 7, [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1)]),
    (TWO_DIM_HISTORY[5], 3, [(1, 2), (1, 3), (1, 4)]),
    (TWO_DIM_HISTORY[6], 3, [(1, 1), (1, 2), (1, 3)]),
])
def test_grid_search_controller_with_two_pruners(history, n_trials, expected_configs):
    history = [TrainLog(FakeConfig(x[0], x[1]), {'automl_metric_val': x[2]}) for x in history]
    expected_configs = [FakeConfig(x[0], x[1]) for x in expected_configs]
    ce = mock.MagicMock()
    ce.estimate.return_value = 0

    gs = GridSearchController(FakeConfig(1, 1), ce, None, [SearchDimension([1, 2, 3, 4], Var1Accessor(), SinglePeakPruner(Var1Accessor())),
                                                           SearchDimension([1, 2, 3, 4], Var2Accessor(), SinglePeakPruner(Var2Accessor()))])
    configs = gs.generate_training_configs(10000, history, n_trials)

    assert configs == expected_configs


@pytest.mark.parametrize("history,n_trials,expected_configs", [
    (TWO_DIM_HISTORY[0], 2, [(1, 1), (3, 1)]),
    (TWO_DIM_HISTORY[1], 1, [(2, 1)]),
    (TWO_DIM_HISTORY[2], 3, [(2, 1), (3, 1), (4, 1)]),
    (TWO_DIM_HISTORY[3], 5, [(3, 1), (4, 1)]),
    (TWO_DIM_HISTORY[4], 4, [(1, 1), (2, 1), (3, 1), (4, 1)]),
    (TWO_DIM_HISTORY[5], 3, [(1, 1), (4, 1)]),
    (TWO_DIM_HISTORY[6], 3, [(1, 1), (2, 1), (3, 1)]),
    (TWO_DIM_HISTORY[7], 3, [(2, 2), (2, 3), (2, 4)]),
])
def test_stage_wise_search_controller(history, n_trials, expected_configs):
    history = [TrainLog(FakeConfig(x[0], x[1]), {'automl_metric_val': x[2]}) for x in history]
    expected_configs = [FakeConfig(x[0], x[1]) for x in expected_configs]
    ce = mock.MagicMock()
    ce.estimate.return_value = 0

    base_config = FakeConfig(1, 1)
    c1 = SingleVarSearchController(base_config, ce, None, [1, 2, 3, 4], Var1Accessor())
    c2 = SingleVarSearchController(base_config, ce, None, [1, 2, 3, 4], Var2Accessor())
    gs = StageWiseSearchController(base_config, [c1, c2])
    configs = gs.generate_training_configs(10000, history, n_trials)

    assert configs == expected_configs


@pytest.mark.parametrize("history,n_trials,expected_configs", [
    (TWO_DIM_HISTORY[0], 2, [(1, 1), (3, 1)]),
    (TWO_DIM_HISTORY[1], 1, [(2, 1)]),
    (TWO_DIM_HISTORY[2], 3, [(2, 1), (3, 1), (4, 1)]),
    (TWO_DIM_HISTORY[3], 5, [(1, 3), (1, 4)]),
    (TWO_DIM_HISTORY[4], 4, [(1, 1), (2, 1), (3, 1), (4, 1)]),
    (TWO_DIM_HISTORY[5], 3, [(4, 1)]),
    (TWO_DIM_HISTORY[6], 3, [(1, 1), (2, 1), (3, 1)]),
    (TWO_DIM_HISTORY[7], 3, [(2, 2), (2, 3), (2, 4)]),
    (TWO_DIM_HISTORY[8], 3, [(2, 3), (2, 4)]),
])
def test_stage_wise_search_controller_with_one_pruner(history, n_trials, expected_configs):
    history = [TrainLog(FakeConfig(x[0], x[1]), {'automl_metric_val': x[2]}) for x in history]
    expected_configs = [FakeConfig(x[0], x[1]) for x in expected_configs]
    ce = mock.MagicMock()
    ce.estimate.return_value = 0

    base_config = FakeConfig(1, 1)
    c1 = SingleVarSearchController(base_config, ce, None, [1, 2, 3, 4], Var1Accessor(), SinglePeakPruner(Var1Accessor()))
    c2 = SingleVarSearchController(base_config, ce, None, [1, 2, 3, 4], Var2Accessor())
    gs = StageWiseSearchController(base_config, [c1, c2])
    configs = gs.generate_training_configs(10000, history, n_trials)

    assert configs == expected_configs


@pytest.mark.parametrize("history,n_trials,expected_configs", [
    (TWO_DIM_HISTORY[0], 2, [(1, 1), (3, 1)]),
    (TWO_DIM_HISTORY[1], 1, [(2, 1)]),
    (TWO_DIM_HISTORY[2], 3, [(2, 1), (3, 1), (4, 1)]),
    (TWO_DIM_HISTORY[3], 5, []),
    (TWO_DIM_HISTORY[4], 4, [(1, 1), (2, 1), (3, 1), (4, 1)]),
    (TWO_DIM_HISTORY[5], 3, [(4, 1)]),
    (TWO_DIM_HISTORY[6], 3, [(1, 1), (2, 1), (3, 1)]),
    (TWO_DIM_HISTORY[7], 3, [(2, 2), (2, 3), (2, 4)]),
    (TWO_DIM_HISTORY[8], 3, []),
])
def test_stage_wise_search_controller_with_two_pruners(history, n_trials, expected_configs):
    history = [TrainLog(FakeConfig(x[0], x[1]), {'automl_metric_val': x[2]}) for x in history]
    expected_configs = [FakeConfig(x[0], x[1]) for x in expected_configs]
    ce = mock.MagicMock()
    ce.estimate.return_value = 0

    base_config = FakeConfig(1, 1)
    c1 = SingleVarSearchController(base_config, ce, None, [1, 2, 3, 4], Var1Accessor(), SinglePeakPruner(Var1Accessor()))
    c2 = SingleVarSearchController(base_config, ce, None, [1, 2, 3, 4], Var2Accessor(), SinglePeakPruner(Var2Accessor()))
    gs = StageWiseSearchController(base_config, [c1, c2])
    configs = gs.generate_training_configs(10000, history, n_trials)

    assert configs == expected_configs


def test_alter_decorator():
    ce = mock.MagicMock()
    ce.estimate.return_value = 0

    original_var2 = 1
    expected_var2 = 3
    base_config = FakeConfig(1, original_var2)
    c1 = SingleVarSearchController(base_config, ce, None, [1, 2, 3, 4], Var1Accessor(), SinglePeakPruner(Var1Accessor()))
    c_a = AlterDecorator(c1, base_config, Var2Accessor(), lambda x: expected_var2)
    for trial in c_a.generate_training_configs(1, [], 2):
        assert Var2Accessor().parse_value(trial) == expected_var2

    best_config = c_a.find_best_config([TrainLog(FakeConfig(1, expected_var2), {'acc': 1})])
    assert best_config.var_2 == original_var2
