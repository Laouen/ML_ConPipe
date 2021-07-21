import graph
import yaml
from graph import Graph

from ConPipe import modules
from ConPipe.utils import find_function_from_modules

# Function to load yaml configuration file
def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config

# TODO: agregar los modulos del usuario
graph_modules = [modules]

def get_node_module(funct_name):
    return find_function_from_modules(
        graph_modules,
        funct_name
    )


class GraphRunner():

    def __init__(self, config):
        self.config = config
        self.verbose = self.config['general']['verbose']
        
        self._load_graph_nodes()

    def _load_graph_nodes(self):

        # Create all DAG nodes
        self.graph_ = Graph()

        for name, config in self.config.items():
            if name != 'general':
                self.graph_.add_node(
                    name, 
                    {
                        **config,
                        'module': get_node_module(config['function'])(
                            config['parameters'],
                            self.verbose
                        )
                    }
                )

        # Build DAG graph
        self.root_nodes_ = []
        for curr_node in self.graph_.nodes():
            input_from = self.graph_.node(curr_node)['input_from']
        
            if len(input_from) == 0:
                self.root_nodes_.append(curr_node)

            for node in input_from:
                self.graph_.add_edge(curr_node, node)
        
    def run(self):

        # TODO: load output from disk

        for curr_node in self.graph_.topological_sort():

            if self.verbose > 1:
                print(f'executing node {curr_node}')
            
            # Collect inputs from dependent nodes
            inputs = {} 
            for node in self.graph_.node(curr_node)['input_from']:
                inputs.update(self.graph_.node(node)['output'])
            
            # Compute output and save result in graph
            output = self.graph_.node(node)['module'].run(**inputs)
            self.graph_.node(node)['output'] = output

            # TODO: save output to disk


