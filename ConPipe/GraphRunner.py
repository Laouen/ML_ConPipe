from pathlib import Path
from graph import Graph
from datetime import datetime
import yaml
import json
import os
import pickle
import numpy as np
import glob
import pandas as pd

from ConPipe.FunctionModule import FunctionModule
from ConPipe.module_loaders import add_path_to_modules, get_class, get_function
from ConPipe.Logger import Logger

# Function to load yaml configuration file
def load_config(config_path, extra_parameters):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    for module, kwargs in extra_parameters.items():
        if 'parameters' in config[module]:
            config[module]['parameters'].update(kwargs)
        else:
            config[module].update(kwargs)

    return config

def bypass_node(*args, **kwargs):

    outputs = {}

    if 'bypass_inout_map' in kwargs:
        for from_input, to_output in kwargs['bypass_inout_map'].items():
            if from_input in kwargs:
                outputs[to_output] = kwargs[from_input]
            elif from_input.isdigit() and (0 < int(from_input) < len(args)):
                outputs[to_output] = args[int(from_input)]
            else:
                raise ValueError('Input {from_input} not in node inputs')
    
    return outputs


class GraphRunner():

    def __init__(self, config_path, extra_parameters):
        self.config = load_config(config_path, extra_parameters)
        self.logger = Logger(self.config['general']['verbose'])
        self.save_dir = os.path.join(
            self.config['general']['save_path'],
            'execution_state'
        )
        self.pandas_sep = ';'
        if 'pandas_sep'in self.config['general']:
            self.pandas_sep = self.config['general']['pandas_sep']

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
            if 'bypass' in config and config['bypass']:
                self.logger(4, f'bypassed node {name}')

                module = FunctionModule(
                    function=bypass_node,
                    parameters=config
                )

            elif 'class' in config:
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

            if 'dependencies' in node:
                for dep_node in node['dependencies']:
                    self.logger(6, f'add dependency {dep_node} to node {node_name}', 1)
                    self.graph_.add_edge(dep_node, node_name)

            if 'input_map' not in node or len(node['input_map']) == 0:
                self.logger(4, f'node {node_name} has no input')
                continue

            self.logger(4, f'create graph dependency connections for node {node_name}')
            for input_node in node['input_map'].keys():
                self.logger(6, f'add input/output dependency {input_node} to node {node_name}', 1)
                self.graph_.add_edge(input_node, node_name)
        
        self._load_nodes_state()
        self._load_nodes_output()

    def _load_nodes_state(self):

        self.logger(2, 'Load node run state')
        for node_name in self.graph_.nodes():
            node = self.graph_.node(node_name)
            run_state_file = os.path.join(
                self.save_dir, 
                node['name'],
                f'run_state.json'
            )
            if os.path.exists(run_state_file):
                node['run_state'] = json.load(open(run_state_file, 'r'))
                node['run_state']['last_run'] = datetime.fromisoformat(node['run_state']['last_run'])
            else:
                node['run_state'] = {'last_run': None}

    def _load_nodes_output(self):

        self.logger(2, 'Load node cached outputs')

        for node_name in self.graph_.nodes():
            node = self.graph_.node(node_name)
            output_dir = os.path.join(self.save_dir, node['name'], 'output')
            if 'cache_output' in node and not node['cache_output']: # The absence of the cache_output parameter means by default cache output
                self.logger(4, f'Node {node_name} cache output set to False', 1)
            elif os.path.isdir(output_dir):    
                self.logger(4, f'Load node {node_name} cached outputs', 1)
                node['output'] = {}
                for file in glob.glob(os.path.join(output_dir, '*.json')):

                    if file.split('/')[-1] == 'run_state.json':
                        continue

                    output_name = file.split('/')[-1].replace('.json', '')
                    self.logger(6, f'Load {node_name}.{output_name}.json output', 2)
                    node['output'][output_name] = json.load(open(file, 'r'))
                
                for file in glob.glob(os.path.join(output_dir, '*.npy')):
                    output_name = file.split('/')[-1].replace('.npy', '')
                    self.logger(6, f'Load {node_name}.{output_name}.npy output', 2)
                    node['output'][output_name] = np.load(file)
                
                for file in glob.glob(os.path.join(output_dir, '*.csv')):
                    output_name = file.split('/')[-1].replace('.csv', '')
                    self.logger(6, f'Load {node_name}.{output_name}.csv output', 2)
                    node['output'][output_name] = pd.read_csv(
                        file,
                        sep=self.pandas_sep
                    )
                
                for file in glob.glob(os.path.join(output_dir, '*.pickle')):
                    output_name = file.split('/')[-1].replace('.pickle', '')
                    self.logger(6, f'Load {node_name}.{output_name}.pickle output', 2)
                    node['output'][output_name] = pickle.load(open(file, 'rb'))

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
                output_file = os.path.join(output_dir, f'{output_name}.json')
                self.logger(4, f'Saving output {output_name} to {output_file}')
                json.dump(output_val, open(output_file, 'w'), indent=2)
            
            elif output_types[output_name] == 'csv':
                output_file = os.path.join(output_dir, f'{output_name}.csv')
                self.logger(4, f'Saving output {output_name} to {output_file}')
                output_val.to_csv(
                    output_file,
                    sep=self.pandas_sep,
                    index=False
                )

            elif output_types[output_name] == 'npy':
                output_file = os.path.join(output_dir, f'{output_name}.npy')
                self.logger(4, f'Saving output {output_name} to {output_file}')
                np.save(output_file, output_val)
            
            elif output_types[output_name] == 'pickle':
                output_file = os.path.join(output_dir, f'{output_name}.pickle')
                self.logger(4, f'Saving output {output_name} to {output_file}')
                pickle.dump(output_val, open(output_file, 'wb'))
            
        # Save the node run state as already run
        run_state_file = os.path.join(output_dir, f'run_state.json')
        json.dump(run_state_file, {'last_run': datetime.now().isoformat()})

    def run(self):

        self.logger(2, 'Run execution graph')
        for node_name in self.graph_.topological_sort():

            self.logger(1, f'Processing node {node_name}')
            node = self.graph_.node(node_name)
            
            # If node is already calculated, then skip recalculation
            if node['output'] is not None:
                self.logger(2, f'Skipping already executed node: {node_name}')
                continue

            if 'force_not_rerun' in node and node['force_not_rerun'] and node['run_state']['last_run'] is not None:
                self.logger(2, f'Skipping already executed note with force not rerun: {node_name}')

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
