import yaml
from graph import Graph

from ConPipe.DataSplit import DataSplit
from ConPipe.ModelSelection import ModelSelection
from ConPipe.ModelEvaluation import ModelEvaluation

# Function to load yaml configuration file
def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


class GraphRunner():

    def __init__(self, config):
        self.config = config

        self._load_graph_nodes()

        self.data_preprocess = DataProcess(self.config['data_preprocess'])
        self.feature_extraction = FeatureExtraction(self.config['feature_extraction'])
        self.data_augmentation = DataAugmentation(self.config['data_augmentation'])

        self.data_split = DataSplit(
            self.config['evaluation_schema'],
            self.config['general']['verbose']
        )
        
        self.model_selection = ModelSelection(
            self.config['model_selection'],
            self.config['general']['verbose']
        )
        
        self.model_evaluation = ModelEvaluation(
            self.config['model_evaluation']
            self.config['general']['verbose']
        )

    def _load_graph_nodes(self):

        # Create all DAG nodes
        self.graph_ = Graph()

        for name, config in self.config.items():
            if name !== 'general':
                self.graph_.add_node(name, {'config': config})

        # Build DAG graph
        self.root_nodes_ = []
        for curr_node in self.graph_nodes_:
            prev_nodes = [
                node 
                for node in self.graph_nodes_ 
                if node['name'] in curr_node['config']['input_from']
            ]
            
            if len(prev_nodes) > 0:
                curr_node.relate(prev_nodes, 'INPUT_FROM')
            else:
                self.root_nodes_.append(curr_node)

    def run()
