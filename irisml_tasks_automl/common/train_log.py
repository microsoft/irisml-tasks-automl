class TrainLog(object):
    """
    Train log, which is a pair of config and its corresponding values under performance metrics.
    """

    def __init__(self, config, metric: dict, automl_metric_name: str = None, time_cost=0, err_msg=None):
        assert config
        assert metric

        self.config = config
        self.metric = metric
        if not automl_metric_name:
            metric_names = metric.keys()
            if len(metric_names) != 1:
                raise ValueError(f'automl_metric_name is missing and there are {len(metric_names)} metrics provided in metric.')
            automl_metric_name = list(metric_names)[0]
        assert automl_metric_name in metric
        self.automl_metric_name = automl_metric_name
        self.time_cost = time_cost
        self.err_msg = err_msg

    def __gt__(self, other):
        return self.automl_metric_val > other.automl_metric_val

    @property
    def automl_metric_val(self):
        return self.metric[self.automl_metric_name]
