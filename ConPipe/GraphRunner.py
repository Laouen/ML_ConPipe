from os import scandir
import yaml
from graph import Graph

from ConPipe.FunctionModule import FunctionModule
from ConPipe.module_loaders import add_path_to_modules, get_class, get_function
from ConPipe.Logger import Logger

# Function to load yaml configuration file
def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config



class GraphRunner():

    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.logger = Logger(self.config['general']['verbose'])

        self.logger(3, 'add paths to modules')
        add_path_to_modules(
            self.config['general']['module_paths'],
            self.logger
        )

        self._load_graph()

    def _load_graph(self):

        self.logger(3, 'Create all DAG nodes')
        self.graph_ = Graph()

        for name, config in self.config.items():
            if name == 'general':
                continue

            # Obtain the module to run
            if 'class' in config:
                self.logger(4, f'Add class node {name} to the execution graph')
                module = get_class(config['class'])(
                    **config['parameters']
                )

            elif 'function' in config:
                self.logger(4, f'Add function node {name} to the execution graph')
                module = FunctionModule(
                    function=get_function(config['function']),
                    parameters=config['parameters'] if 'parameters' in config else {}
                )

            else:
                raise AttributeError('Either a class or a function module must be specified')

            self.graph_.add_node(
                name, 
                {
                    **config,
                    'module': module,
                    'output': None
                }
            )

        self.logger(3, 'Build DAG graph')
        for node_name in self.graph_.nodes():

            node = self.graph_.node(node_name)

            if 'input_from' not in node or len(node['input_from']) == 0:
                self.logger(4, f'node {node_name} has no input')
                continue

            self.logger(4, f'create graph dependency connections for node {node_name}')
            for input_node in node['input_from']:
                self.logger(6, f'\tadd dependency {input_node} to node {node_name}')
                self.graph_.add_edge(node_name, input_node)


    def run(self):

        # TODO: load output from disk
        # TODO: make a system to restart training from last place

        self.logger(2, 'Run execution graph')
        for node_name in self.graph_.topological_sort():

            self.logger(1, f'Processing node {node_name}')
            node = self.graph_.node(node_name)
            
            # If node is already calculated, then skip recalculation
            if node['output'] is not None:
                self.logger(2, f'Skipping already executed node: {node_name}')
                continue

            self.logger(4, f'Collect {node_name} inputs from dependent nodes')
            args = []
            kwargs = {} 
            for sender_node, input_map in node['input_map'].items():
                self.logger(6, f'\tCollect output from {sender_node}')
                output = self.graph_.node(sender_node)['output']
                for from_param, to_param in input_map.items():
                    self.logger(10, f'\tMap output {sender_node}.{from_param} output to {node_name}.{to_param} input')
                    
                    if type(to_param) == int:
                        args.append((to_param, output[from_param]))
                    else:
                        kwargs[to_param] = output[from_param]
            args = sorted(args, key=lambda x: x[0])
            args = [x[1] for x in args]
            
            self.logger(2, f'Executing {node_name}')
            node['output'] = node['module'].run(*args, **kwargs)

            # TODO: save output to disk


