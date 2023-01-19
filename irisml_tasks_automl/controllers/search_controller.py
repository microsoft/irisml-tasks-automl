from abc import ABC, abstractmethod
from typing import List
from .search_pruners import CandidatePruner
from ..common.base_config import ConfigVarAccessor
from ..common.train_log import TrainLog
import random

from copy import deepcopy


class BaseAutomlController(ABC):
    """
    Base class defining general automl controller, that generate new configs to try given history and budget
    """

    def __init__(self, cost_estimator, base_config):
        self.base_config = base_config
        self.cost_estimator = cost_estimator

    @abstractmethod
    def generate_training_configs(self, budget_in_secs: int, history: List[TrainLog], n_trials: int):
        """generate n_trials training configs at most to try in the next round within budget limit being budget_in_secs, based on previous training history
        Args:
            budget_in_secs: time budget in seconds
            history: training history
            n_trials: max number of configs to try in the next round

        Returns:
            a collection of training configs
        """
        pass

    def set_base_config(self, config):
        self.base_config = deepcopy(config)

    def find_best_config(self, history: List[TrainLog]):
        if not history:
            return None

        return max(history).config


class SearchDimension(object):
    def __init__(self, candidates, var_accessor: ConfigVarAccessor, pruner: CandidatePruner = None, candidates_order=None):
        self.candidates = candidates
        self.var_accessor = var_accessor
        self.pruner = pruner
        if candidates_order:
            assert len(candidates) == len(candidates_order)
            for x in candidates:
                assert x in candidates_order

        self.candidates_order = candidates_order or candidates


class SingleVarSearchController(BaseAutomlController):
    """
    A controller that searches one dimension/variable in config, to find the best config in a heuristic manner
    """

    def __init__(self, base_config, cost_estimator, dataset, candidates, var_accessor, pruner=None, random_seed=None, candidates_order=None):
        super(SingleVarSearchController, self).__init__(cost_estimator, base_config)
        self.dataset = dataset
        self.search_dim = SearchDimension(candidates, var_accessor, pruner, candidates_order)
        self.random_seed = random_seed

    def generate_training_configs(self, budget_in_secs, history, n_trials):
        used_budget = 0
        result = []
        candidates_in_order = [x for x in self.search_dim.pruner.prune(self.base_config, self.search_dim.candidates_order, history)] if self.search_dim.pruner else self.search_dim.candidates
        candidates = [x for x in self.search_dim.candidates if x in candidates_in_order]
        candidate_configs = [self.search_dim.var_accessor.assign_val_to_config(self.base_config, x) for x in candidates]
        partial_history = self.keep_history_varied_from_base_config(history)
        partial_history_configs = [x.config for x in partial_history]

        if self.random_seed:
            random.Random(self.random_seed).shuffle(candidate_configs)

        for candidate_config in candidate_configs:
            if len(result) >= n_trials:
                return result

            if candidate_config in partial_history_configs:
                continue

            cost = self.cost_estimator.estimate(candidate_config, self.dataset)
            if cost > budget_in_secs - used_budget:
                continue

            used_budget += cost
            result.append(candidate_config)

        return result

    def keep_history_varied_from_base_config(self, history: List[TrainLog]):
        if not history:
            return []
        var_accessor = self.search_dim.var_accessor
        return [x for x in history if var_accessor.assign_val_to_config(x.config, var_accessor.parse_value(self.base_config)) == self.base_config]

    def find_best_config(self, history: List[TrainLog]):
        history = self.keep_history_varied_from_base_config(history)
        if not history:
            return None
        return super(SingleVarSearchController, self).find_best_config(history)


class GridSearchController(BaseAutomlController):
    """
    A controller that searches across different dimensions/variables in config in a grid search manner, to find the best config in a heuristic manner, it stops generating, if
    - reaching the number of desired configs or no more configs worth trying

    grid search: https://en.wikipedia.org/wiki/Hyperparameter_optimization
    """

    def __init__(self, base_config, cost_estimator, dataset, grid_search_dims: List[SearchDimension], random_seed=None):
        super(GridSearchController, self).__init__(cost_estimator, base_config)
        self.search_dims = grid_search_dims
        self.dataset = dataset
        self.random_seed = random_seed

    @staticmethod
    def create_from_single_var_controllers(base_config, cost_estimator, dataset, single_var_controllers: List[SingleVarSearchController], random_seed=None):
        grid_search_dims = [c.search_dim for c in single_var_controllers]
        return GridSearchController(base_config, cost_estimator, dataset, grid_search_dims, random_seed)

    def generate_training_configs(self, budget_in_secs, history, n_trials):
        return self.grid_search_configs(0, self.base_config, [], budget_in_secs, n_trials, history)

    def grid_search_configs(self, c_idx, base_config, result, budget, n_trials, history):
        if len(result) >= n_trials or c_idx == len(self.search_dims) or budget <= 0:
            return result

        dim_searcher = self.search_dims[c_idx]
        if c_idx == len(self.search_dims) - 1:
            used_budget = sum([self.cost_estimator.estimate(x, self.dataset) for x in result])
            candidate_configs = [dim_searcher.var_accessor.assign_val_to_config(base_config, x) for x in dim_searcher.candidates]
            history_configs = [x.config for x in history]

            if self.random_seed:
                random.Random(self.random_seed).shuffle(candidate_configs)

            for candidate_config in candidate_configs:
                if len(result) >= n_trials:
                    break

                if candidate_config in history_configs:
                    continue

                cost = self.cost_estimator.estimate(candidate_config, self.dataset)
                if cost > budget - used_budget:
                    continue

                if not all([(not d.pruner) or d.pruner.is_valuable(candidate_config, d.var_accessor.parse_value(candidate_config), d.candidates, history) for d in self.search_dims]):
                    continue

                used_budget += cost
                result.append(candidate_config)
        else:
            for candidate in dim_searcher.candidates:
                config = dim_searcher.var_accessor.assign_val_to_config(base_config, candidate)
                result = self.grid_search_configs(c_idx + 1, config, result, budget, n_trials, history)

        return result


class StageWiseSearchController(BaseAutomlController):
    """
    A controller that searches across different dimensions/variables in config, to find the best config in a heuristic manner, it stops generating, if
    1. reaching the number of desired configs or no more configs worth trying
    2. if dimension n is not exhausted, even less generating less than n_trials, it will not proceed to dimension n+1

    heuristic:
    - dimension 1: base_config passed in will be used as the base
    - dimension n: the best config (with highest performance) in dimension n-1 will be used as the base
    """

    def __init__(self, base_config, controllers: List[BaseAutomlController]):
        super(StageWiseSearchController, self).__init__(None, base_config)
        self.single_var_controllers = controllers

    def generate_training_configs(self, budget_in_secs, history, n_trials):
        base_config = deepcopy(self.base_config)
        for controller in self.single_var_controllers:
            controller.set_base_config(base_config)
            candidates = controller.generate_training_configs(budget_in_secs, history, n_trials)

            if candidates:
                # don't proceed to next stage, if current stage is not finished
                return candidates

            base_config = controller.find_best_config(history) or base_config

        return []


class AlterDecorator(BaseAutomlController):
    """
    A helper decorator that temporarily modified the base config within currently controller
    Example: when trying horizontally flip, most likely 5 epochs is good enough, and we could apply this decorator to HFlipController to temporarily change the epoch number to 5
    """

    def __init__(self, controller, base_config, var_accessor: ConfigVarAccessor, alter_func):
        self.backup_val = var_accessor.parse_value(base_config)
        self.var_accessor = var_accessor
        self.alter_func = alter_func
        self.controller = controller
        self.controller.set_base_config(self._alter_config(base_config))
        super().__init__(self.controller.cost_estimator, self.controller.base_config)

    def generate_training_configs(self, budget_in_secs: int, history: List[TrainLog], n_trials: int):
        return self.controller.generate_training_configs(budget_in_secs, history, n_trials)

    def find_best_config(self, history: List[TrainLog]):
        config = self.controller.find_best_config(history)
        if not config:
            return None
        return self.var_accessor.assign_val_to_config(config, self.backup_val)

    def set_base_config(self, config):
        self.controller.set_base_config(self._alter_config(config))

    def _alter_config(self, config):
        return self.var_accessor.assign_val_to_config(config, self.alter_func(config))
