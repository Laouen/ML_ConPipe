from pathlib import Path
import yaml
from graph import Graph
import json
import os
import pickle
import numpy as np

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
        self.save_dir = os.path.join(
            self.config['general']['save_path'],
            'execution_state'
        )

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

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

            parameter = config['parameters'] if 'parameters' in config else {}

            # Obtain the module to run
            if 'class' in config:
                self.logger(4, f'Add class node {name} to the execution graph')
                module = get_class(config['class'])(
                    **parameter
                )

            elif 'function' in config:
                self.logger(4, f'Add function node {name} to the execution graph')
                module = FunctionModule(
                    function=get_function(config['function']),
                    parameters=parameter
                )

            else:
                raise AttributeError('Either a class or a function module must be specified')

            self.graph_.add_node(
                name, 
                {
                    **config,
                    'module': module,
                    'name': name,
                    'output': None
                }
            )

        self.logger(3, 'Build DAG graph')
        for node_name in self.graph_.nodes():

            node = self.graph_.node(node_name)

            if 'input_map' not in node or len(node['input_map']) == 0:
                self.logger(4, f'node {node_name} has no input')
                continue

            self.logger(4, f'create graph dependency connections for node {node_name}')
            for input_node in node['input_map'].keys():
                self.logger(6, f'add dependency {input_node} to node {node_name}', 1)
                self.graph_.add_edge(input_node, node_name)

    def _load_nodes_inputs(self):

        for node_name in self.graph_.nodes():
            node = self.graph_.node(node_name)

            
    
    def _save_output(self, node):

        self.logger(2, f'Saving {node["name"]} output')

        # Create the node folder where to store output
        output_dir = os.path.join(self.save_dir, node['name'], 'output')
        Path(output_dir).mkdir(
            parents=True, 
            exist_ok=True
        )

        self.logger(6, f'Saving output to {output_dir}')

        output_types = node['output_storage_type']

        if type(output_types) == str:
            output_types = {
                output_name: output_types
                for output_name in node['output'].keys()
            }

        for output_name, output_val in node['output'].items():
            if output_types[output_name] == 'json':
                output_file = open(os.path.join(output_dir, f'{output_name}.json'), 'w')
                self.logger(6, f'Saving output {output_name} to {output_file}')
                json.dump(output_val, output_file, indent=2)
            
            elif output_types[output_name] == 'csv':
                output_file = os.path.join(output_dir, f'{output_name}.csv')
                self.logger(6, f'Saving output {output_name} to {output_file}')
                output_val.to_csv(output_file, sep=';', index=False)

            elif output_types[output_name] == 'npy':
                output_file = os.path.join(output_dir, f'{output_name}.npy')
                self.logger(6, f'Saving output {output_name} to {output_file}')
                np.save(output_file, output_val)
            
            elif output_types[output_name] == 'pickle':
                output_file = os.path.join(output_dir, f'{output_name}.pickle')
                self.logger(6, f'Saving output {output_name} to {output_file}')
                pickle.dump(output_val, open(output_file, 'wb'))


    def run(self):

        # TODO: load output from disk
        # TODO: make a system to restart training from last place

        print(self.graph_.to_dict())

        self.logger(2, 'Run execution graph')
        for node_name in self.graph_.topological_sort():

            self.logger(1, f'Processing node {node_name}')
            node = self.graph_.node(node_name)
            
            # If node is already calculated, then skip recalculation
            if node['output'] is not None:
                self.logger(2, f'Skipping already executed node: {node_name}')
                continue

            args = []
            kwargs = {}
            if 'input_map' in node:
                self.logger(4, f'Collect {node_name} inputs from dependent nodes') 
                for sender_node, input_map in node['input_map'].items():
                    self.logger(6, f'Collect input from {sender_node}', 1)
                    output = self.graph_.node(sender_node)['output']
                    for from_param, to_param in input_map.items():
                        self.logger(10, f'Map {sender_node}.{from_param} output to {node_name}.{to_param} input', 2)
                        
                        if type(to_param) == int:
                            args.append((to_param, output[from_param]))
                        else:
                            kwargs[to_param] = output[from_param]
                args = sorted(args, key=lambda x: x[0])
                args = [x[1] for x in args]
            
            self.logger(2, f'Executing {node_name}')
            node['output'] = node['module'].run(*args, **kwargs)

            if 'output_storage_type' in node:
                self._save_output(node)
