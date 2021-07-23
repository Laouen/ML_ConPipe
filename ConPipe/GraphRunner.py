from os import scandir
import yaml
from graph import Graph

from ConPipe.FunctionModule import FunctionModule
from ConPipe.ModuleLoader import ModuleLoader
from ConPipe.Logger import Logger
from ConPipe.DefaultModules import DEFAULT_MODULES

# Function to load yaml configuration file
def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config



class GraphRunner():

    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.logger = Logger(self.config['general']['verbose'])
        self.loader = ModuleLoader(
            self.config['general']['module_directories'],
            self.config['general']['installed_modules']
        )
        self._load_graph()

    def _load_graph(self):

        # Create all DAG nodes
        self.graph_ = Graph()

        for name, config in self.config.items():
            if name == 'general':
                continue

            # Obtain the module to run
            if 'class' in config:
                module = self.loader.get_class(**config['class'])(
                    **config['parameters']
                )

            elif 'function' in config:
                module = FunctionModule(
                    function=self.loader.get_function(**config['function']),
                    parameters=config['parameters']
                )

            else:
                raise AttributeError('Either a class or a function module must be specified')

            self.graph_.add_node(
                name, 
                {**config, 'module': module}
            )

        # Build DAG graph
        for curr_node_name in self.graph_.nodes():

            curr_node = self.graph_.node(curr_node_name)

            if 'input_from' not in curr_node or len(curr_node['input_from']) == 0:
                continue

            for node in curr_node['input_from']:
                self.graph_.add_edge(curr_node_name, node)


    def run(self):

        # TODO: load output from disk
        # TODO: make a system to restart training from last place

        for curr_node_name in self.graph_.topological_sort():

            curr_node = self.graph_.node(curr_node_name)
            
            # If node is already calculated, then skip recalculation
            if curr_node['output'] is not None:
                self.logger(1, f'Skipping already executed node: {curr_node}')
                continue

            self.logger(1, f'executing node: {curr_node}')
            
            # Collect inputs from dependent nodes
            inputs = {} 
            for node in self.graph_.node(curr_node)['input_from']:
                inputs.update(self.graph_.node(node)['output'])
            
            # Compute output and save result in graph
            output = self.graph_.node(node)['module'].run(**inputs)
            self.graph_.node(node)['output'] = output

            # TODO: save output to disk


