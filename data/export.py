import argparse

from data.construct import load_data_from_configuration
import yaml


def set_config_value(cfg, key, value):
    """ sets the value of a key in the configuration using `.` to access deeper levels. """
    path = key.split('.')
    for p in path[:-1]:
        cfg = cfg[p]
    cfg[path[-1]] = value


def export(args):
    """ exports dataset splits to a pickle file. """
    with open(args.config) as f:
        config = yaml.safe_load(f)
    assert len(args.override) % 2 == 0, f'Always specify pairs of keys and values (received {len(args.override)} tokens.'
    for i in range(0, len(args.override), 2):
        set_config_value(config, args.override[2 * i], args.override[2 * i + 1])
    print(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export datasets used in this project as pickle files.')
    parser.add_argument('config', help='Experiment configuration as .yaml file.')
    parser.add_argument('output', help='Output file for all the splits')
    parser.add_argument('-o','--override', dest='override', nargs='*', help='Pairs of arguments {key} {value} to override in the configuration. Use `.` to index deeper levels of the configuration.')
    export(parser.parse_args)