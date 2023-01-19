from irisml_tasks_automl import DictConfigVarAccessor, DictBasedConfig
import pytest


@pytest.mark.parametrize("path,values", [
    ('optim/base_lr', [3, 4]),
    ('optim', ['wow', 'lol']),
    ('optim/lr/stage_iters', [{'1': 1}, {'2': 2}]),
])
def test_dict_config_assign_with_another_config(path, values):
    print(path)
    config1 = DictBasedConfig([path])
    config2 = DictBasedConfig([path])

    accessor = DictConfigVarAccessor(path)

    config1 = accessor.assign_val_to_config(config1, values[0])
    config2 = accessor.assign_val_to_config(config2, values[1])
    config3 = accessor.assign_val_to_config(config2, accessor.parse_value(config1))

    assert accessor.parse_value(config3) == accessor.parse_value(config1)
    assert accessor.parse_value(config3) == values[0]


@pytest.mark.parametrize("path,val", [
    ('optim/base_lr', 3),
    ('optim', "wow"),
    ('optim/lr/stage_iters', {'1': 1}),
])
def test_dict_config_assign_value(path, val):
    config = DictBasedConfig([path])

    accessor = DictConfigVarAccessor(path)
    config = accessor.assign_val_to_config(config, val)
    assert accessor.parse_value(config) == val
