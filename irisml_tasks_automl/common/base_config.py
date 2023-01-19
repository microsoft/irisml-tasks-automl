from abc import ABC, abstractmethod
from copy import deepcopy


class FlexibleBaseConfig(ABC):
    """
    A flexible base class for config.
    """

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               len(self.__dict__) == len(other.__dict__) and \
               all([key in other.__dict__ and self.__dict__[key] == other.__dict__[key] for key in self.__dict__])


class DictBasedConfig(dict):
    """
    A base class for dictionary-based config, with constructor taking paths for defining a config
    Note that a naive dict would suffice as a dict-based config. This class adds an additional path-based way for initialization
    """

    SEPARATOR = '/'

    def __init__(self, paths=None):
        super().__init__()
        if paths:
            self.add_paths(paths)

    def add_paths(self, paths):
        if not paths or not isinstance(paths, list):
            raise RuntimeError('paths is not list.')

        for path in paths:
            temp = self
            parts = path.split(DictBasedConfig.SEPARATOR)
            for part in parts:
                if part not in temp:
                    temp[part] = {}
                temp = temp[part]


class ConfigVarAccessor(ABC):
    """
    accessor for certain dimension/variables in config
    """

    @abstractmethod
    def assign_val_to_config(self, config, val):
        pass

    @abstractmethod
    def parse_value(self, config):
        pass


class DictConfigVarAccessor(ConfigVarAccessor):
    """
    accessor for certain dimension/variables in dictionary (i.e. dict()) (not working for FlexibleBaseConfig)
    """

    def __init__(self, path):
        self._paths = path.split(DictBasedConfig.SEPARATOR)

    def assign_val_to_config(self, config: dict, val):
        DictConfigVarAccessor._throw_if_not_dict(config)

        result = deepcopy(config)
        last_level = self._paths[-1]
        self._access_second_to_the_last_level(result)[last_level] = val
        return result

    def parse_value(self, config: dict):
        DictConfigVarAccessor._throw_if_not_dict(config)

        last_level = self._paths[-1]
        return self._access_second_to_the_last_level(config)[last_level]

    def _access_second_to_the_last_level(self, config: dict):
        temp = config
        for i, path in enumerate(self._paths):
            if i >= len(self._paths) - 1:
                break

            temp = temp[path]

        return temp

    @staticmethod
    def _throw_if_not_dict(config):
        if not config or not isinstance(config, dict):
            raise RuntimeError('config is not of type dict.')
