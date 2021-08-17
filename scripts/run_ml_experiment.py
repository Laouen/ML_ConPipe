from ConPipe.GraphRunner import GraphRunner
import argparse
import numpy as np
from collections import defaultdict
from time import sleep


def main(config_path, extra_parameters):
    graph = GraphRunner(config_path, extra_parameters)
    graph.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a Machine Learning experiment from a YAML graph representation of the function flows')

    parser.add_argument(
        'config_path',
        type=str,
        help='The path to the yaml config file to run'
    )

    parser.add_argument('extra_parameters', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    n_extra_params = len(args.extra_parameters)
    if n_extra_params % 2 != 0:
        raise ValueError('The number of extra parameters must be even because they are pairs of --<param_name> <param_value>')

    paired_params = np.array(args.extra_parameters).reshape(n_extra_params // 2, 2)
    extra_parameters = defaultdict(lambda: dict())
    for (parameter, value) in paired_params:
        module, parameter_name = parameter.split('.')
        module = module.strip('-')
        extra_parameters[module].update({parameter_name: value})

    main(args.config_path, extra_parameters)