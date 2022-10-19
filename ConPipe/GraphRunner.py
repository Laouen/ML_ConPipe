from pathlib import Path
from graph import Graph
from datetime import datetime
import hiyapyco
import json
import os
import copy
import pickle
import numpy as np
import glob
import pandas as pd

from ConPipe.FunctionModule import FunctionModule
from ConPipe.ModuleLoader import add_path_to_modules, get_class, get_function
from ConPipe.Logger import Logger


class GraphRunner():

    def __init__(self, config_paths, custom_config):

        if len(config_paths) == 0:
            raise ValueError('There must be at least one config path')

        config_path_folders = [
            str(Path(p).parent.resolve())
            for p in config_paths
        ]
        if len(set(config_path_folders)) > 1:
            raise ValueError('All the config files must be'
                             'placed in the same directory')

        self.config = GraphRunner._load_config(
            config_paths,
            custom_config
        )
        
        general_config = self.config.get('general', {})

        self.logger = Logger(general_config.get('verbose', 1))
        self.pandas_sep = general_config.get('pandas_sep', ';')
        self.root_path = str(Path(config_paths[0]).parent.resolve())

        # Calculate save path where to store the nodes outputs and pipeline state
        save_path = general_config.get('save_path', './')
        if not os.path.isabs(save_path):
            save_path = os.path.join(self.root_path, save_path)

        self.save_dir = os.path.join(
            save_path,
            'execution_state'
        )
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.logger(3, 'add paths to modules')
         
        add_path_to_modules(
            self.root_path,
            general_config.get('modules_root', ''),
            general_config.get('module_paths', []),
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
            if config.get('bypass', False):
                self.logger(4, f'bypassed node {name}')

                module = FunctionModule(
                    function=GraphRunner._bypass_node,
                    parameters=config
                )

            elif 'class' in config:
                self.logger(4, f'Add class node {name} to the execution graph')
                module = get_class(config['class'])(
                    **config.get('parameters', {})
                )

            elif 'function' in config:
                self.logger(4, f'Add function node {name} to the execution graph')
                module = FunctionModule(
                    function=get_function(config['function']),
                    parameters=config.get('parameters', {})
                )

            else:
                raise AttributeError(
                    'Either a class a function module must be specified'
                    'or the module must be set to bypass = True'
                )

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

            for dep_node in node.get('dependencies', []):
                self.logger(6, f'add dependency {dep_node} to node {node_name}', 1)
                self.graph_.add_edge(dep_node, node_name)

            for input_node in node.get('input_map',{}).keys():
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
                with open(run_state_file, 'r', encoding='utf-8') as file:
                    node['run_state'] = json.load(file)
                    node['run_state']['last_run'] = datetime.fromisoformat(node['run_state']['last_run'])
            else:
                node['run_state'] = {'last_run': None}

    def _load_nodes_output(self):

        self.logger(2, 'Load node cached outputs')

        for node_name in self.graph_.nodes():
            node = self.graph_.node(node_name)
            output_dir = os.path.join(self.save_dir, node['name'], 'output')
            if not node.get('cache_output', True):
                self.logger(4, f'Node {node_name} cache output set to False', 1)
            
            elif os.path.isdir(output_dir):    
                self.logger(4, f'Load node {node_name} cached outputs', 1)
                node['output'] = {}
                for file in glob.glob(os.path.join(output_dir, '*.json')):

                    if file.split('/')[-1] == 'run_state.json':
                        continue

                    output_name = file.split('/')[-1].replace('.json', '')
                    self.logger(6, f'Load {node_name}.{output_name}.json output', 2)
                    with open(file, 'r', encoding='utf-8') as json_file:
                        node['output'][output_name] = json.load(json_file)
                
                for file in glob.glob(os.path.join(output_dir, '*.npy')):
                    output_name = file.split('/')[-1].replace('.npy', '')
                    self.logger(6, f'Load {node_name}.{output_name}.npy output', 2)
                    node['output'][output_name] = np.load(file)
                
                for file in glob.glob(os.path.join(output_dir, '*.csv')):
                    output_name = file.split('/')[-1].replace('.csv', '')
                    self.logger(6, f'Load {node_name}.{output_name}.csv output', 2)
                    node['output'][output_name] = pd.read_csv(
                        file,
                        sep=self.pandas_sep,
                        encoding='utf-8'
                    )
                
                for file in glob.glob(os.path.join(output_dir, '*.pickle')):
                    output_name = file.split('/')[-1].replace('.pickle', '')
                    self.logger(6, f'Load {node_name}.{output_name}.pickle output', 2)
                    with open(file, 'rb') as pickle_file:
                        node['output'][output_name] = pickle.load(pickle_file)

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
                with open(output_file, 'w', encoding='utf-8') as json_file:
                    json.dump(output_val, json_file, indent=2)
            
            elif output_types[output_name] == 'csv':
                output_file = os.path.join(output_dir, f'{output_name}.csv')
                self.logger(4, f'Saving output {output_name} to {output_file}')
                output_val.to_csv(
                    output_file,
                    sep=self.pandas_sep,
                    index=False,
                    encoding='utf-8'
                )

            elif output_types[output_name] == 'npy':
                output_file = os.path.join(output_dir, f'{output_name}.npy')
                self.logger(4, f'Saving output {output_name} to {output_file}')
                np.save(output_file, output_val)
            
            elif output_types[output_name] == 'pickle':
                output_file = os.path.join(output_dir, f'{output_name}.pickle')
                self.logger(4, f'Saving output {output_name} to {output_file}')
                with open(output_file, 'wb') as pickle_file:
                    pickle.dump(output_val, pickle_file)
            
        # Save the node run state as already run
        run_state_path = os.path.join(output_dir, f'run_state.json')
        with open(run_state_path, 'w', encoding='utf-8') as state_file:
            json.dump({'last_run': datetime.now().isoformat()}, state_file)

    def run(self):
    
        self.logger(2, 'Run execution graph')
        for node_name in self.graph_.topological_sort():

            self.logger(1, f'Processing node {node_name}')
            node = self.graph_.node(node_name)
            
            # If node is already calculated, then skip recalculation
            if node['output'] is not None:
                self.logger(2, f'Skipping already executed node: {node_name}')
                continue

            if node.get('force_not_rerun', False) and node['run_state']['last_run'] is not None:
                self.logger(2, f'Skipping already executed note with force not rerun: {node_name}')

            args = []
            kwargs = {}
            if 'input_map' in node:
                self.logger(4, f'Collect {node_name} inputs from dependent nodes') 
                for sender_node, input_map in node['input_map'].items():
                    self.logger(6, f'Collect input from {sender_node}', 1)
                    output = self.graph_.node(sender_node)['output']
                    for to_param, from_param in input_map.items():
                        self.logger(
                            10, f'Map {sender_node}.{from_param} output '
                            f'to {node_name}.{to_param} input', 2
                        )

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

    ######################
    ### Static methods ###
    ######################

    # Function to load yaml configuration file
    @staticmethod
    def _load_config(config_paths, custom_config):

        default_modules = hiyapyco.load(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'default_configurations.yaml'
            ),
            usedefaultyamlloader=True,
            encoding='utf-8'
        )

        config = hiyapyco.load(
            *config_paths,
            method=hiyapyco.METHOD_MERGE,
            mergelists=False,
            interpolate=False,
            usedefaultyamlloader=True,
            encoding='utf-8',
            failonmissingfiles=True
        )

        # Use default modules to complete the config
        for module_name, module_config in config.items():
            if module_config.get('base_module', None) is not None:
                
                if module_name not in default_modules.keys():
                    raise ValueError(
                        f'Base module {module_name} non existent, options are:'
                        f'\n\t' + '\n\t'.join(default_modules.keys())
                    )
                
                config[module_name] = copy.deepcopy(default_modules[module_name])
                GraphRunner.dict_merge(config[module_name], module_config)

        for module_name, kwargs in custom_config.items():
            
            if module_name not in config:
                raise ValueError(
                    f'Custom config module {module_name}'
                    'does not exists'
                )

            GraphRunner.dict_merge(config[module_name], kwargs)

        return config

    # Function used to assign to a node in order to bypass its calculations
    @staticmethod
    def _bypass_node(*args, **kwargs):
    
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

    @staticmethod
    def dict_merge(dct, merge_dct):
        """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
        updating only top-level keys, dict_merge recurses down into dicts nested
        to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
        ``dct``.
        :param dct: dict onto which the merge is executed
        :param merge_dct: dct merged into dct
        :return: None
        """
        for k, v in merge_dct.items():
            if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):
                GraphRunner.dict_merge(dct[k], merge_dct[k])
            else:
                dct[k] = merge_dct[k]
