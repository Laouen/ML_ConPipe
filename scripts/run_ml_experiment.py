from ConPipe.GraphRunner import GraphRunner
import argparse


def main(config_path):
    graph = GraphRunner(config_path)
    graph.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run a Machine Learning experiment from a YAML graph representation of the function flows')

    parser.add_argument(
        'config_path',
        type=str,
        help='The path to the yaml config file to run'
    )

    args = parser.parse_args()

    main(args.config_path)