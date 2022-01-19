from util import format_name
from itertools import product
from copy import deepcopy
import configuration
import logging
import traceback

pipeline_members = {}

def pipeline_log(string):
    logging.info(f'EVALUATION PIPELINE - {string}')

class PipelineMember:
    """ Superclass for all pipeline members. """

    def __init__(self, name=None, log_plots=True, model_kwargs={}, model_kwargs_fit=None, model_kwargs_evaluate=None, **kwargs):
        self._name = name
        self.model_kwargs = model_kwargs
        if model_kwargs_fit is None:
            model_kwargs_fit = self.model_kwargs
        if model_kwargs_evaluate is None:
            model_kwargs_evaluate = self.model_kwargs
        self.model_kwargs_fit = model_kwargs_fit
        self.model_kwargs_evaluate = model_kwargs_evaluate
        self.log_plots = log_plots

    @property
    def prefix(self):
        if self._name is None:
            return f''
        else:
            return f'{self._name}_'
    
    @property
    def suffix(self):
        if self._name is None:
            return f''
        else:
            return f'_{self._name}'

    @property
    def print_name(self):
        if self._name is None:
            return f'{self.name} (unnamed)'
        else:
            return f'{self.name} : "{self._name}"'
    
    @property
    def configuration(self):
        config = {
            'Kwargs to model calls' : self.model_kwargs,
            'Log plots' : self.log_plots,
        }
        if self.model_kwargs_fit != self.model_kwargs:
            config['Kwargs to model calls (fit)'] = self.model_kwargs_fit
        if self.model_kwargs_evaluate != self.model_kwargs:
            config['Kwargs to model calls (evaluate)'] = self.model_kwargs_evaluate
        return config

    def __str__(self):
        return '\n'.join([self.print_name] + [f'\t{key} : {value}' for key, value in self.configuration.items()])

def register_pipeline_member(cls):
    if not issubclass(cls, PipelineMember):
        raise ValueError(f'Trying to register class {cls} which does not subclass {PipelineMember}')
    pipeline_members[cls.name.lower()] = cls
    return cls


def pipeline_configs_from_grid(config, delimiter=':'):
    """ Builds a list of pipeline configs from a grid configuration. 
    
    Parameters:
    -----------
    config : dict
        The pipeline member config.
    
    Returns:
    --------
    configs : list
        A list of pipeline member configs from the grid or just the single config, if no grid was specified.
    """
    config = dict(config).copy() # To avoid errors?
    grid = config.pop('pipeline_grid', None)
    if grid is None:
        if 'name' in config:
            config['name'] = format_name(config['name'], config.get('name_args', []), config)
        return [config]
    else:
        parameters = grid # dict of lists
        keys = list(parameters.keys())
        configs = []
        for values in product(*[parameters[k] for k in keys]):
            subconfig = deepcopy(config)
            for idx, key in enumerate(keys):
                path = key.split(delimiter)
                target = subconfig
                for x in path[:-1]:
                    target = target[x]
                target[path[-1]] = values[idx]
            if 'name' in subconfig:
                subconfig['name'] = format_name(subconfig['name'], subconfig.get('name_args', []), subconfig, delimiter=delimiter)
            configs.append(subconfig)
        return configs


class Pipeline:
    """ Pipeline for stuff to do after a model has been trained. """

    def __init__(self, members: list, config: configuration.EvaluationConfiguration, gpus=0, ignore_exceptions=False):

        self.members = []
        self.ignore_exceptions = ignore_exceptions
        self.config = config
        idx = 0
        for idx, entry in enumerate(members):
            configs = pipeline_configs_from_grid(entry)
            for member in configs:
                # Update settings that the member does not specify with the master configuration
                if 'log_plots' not in member:
                    member['log_plots'] = config.log_plots
                if member['type'].lower() not in pipeline_members:
                    raise ValueError(f'Unspported pipeline class {member["type"]}')
                self.members.append(pipeline_members[member['type'].lower()](
                    gpus = gpus,
                    pipeline_idx = idx,
                    **member,
                ))
                idx += 1

    def __call__(self, *args, **kwargs):
        for member in self.members:
            try:
                pipeline_log(f'Running {member.print_name}...')
                args, kwargs = member(*args, **kwargs)
            except Exception as e:
                pipeline_log(f'{member.print_name} FAILED. Reason: "{e}"')
                print(traceback.format_exc())
                if not self.ignore_exceptions:
                    raise e
        return args, kwargs

    def __str__(self):
        return '\n'.join([
            'Evaluation Pipeline',
        ] + [
            f'{member}' for member in self.members
        ])