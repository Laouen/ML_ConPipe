import sys
import os
from pathlib import Path
import importlib
import inspect

from ConPipe.exceptions import NotFunctionModuleError, NotClassModuleError

def split_module_name(name):
    module = '.'.join(name.split('.')[:-1])
    obj = name.split('.')[-1]
    return module, obj

def get_module_object(object_name):
    module_name, obj_name = split_module_name(object_name)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, obj_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f'El módulo {module_name} no existe')
    except AttributeError:
        raise AttributeError(
            f'El módulo {module_name} no tiene definido a {obj_name}')

def get_function(function_name):
    func = get_module_object(function_name)

    if not hasattr(func, '__call__'):
        raise NotFunctionModuleError(f'{function_name} is not callable (i.e. a function)')
    
    return func

def get_class(class_name):
    class_obj = get_module_object(class_name)

    if not inspect.isclass(class_obj):
        raise NotClassModuleError(f'{class_name} is not a class')
    
    return class_obj

def add_path_to_modules(configs_path, modules_root, module_paths, logger):
    
    if not os.path.isabs(modules_root):
        modules_root = os.path.join(configs_path, modules_root)

    for module_path in module_paths:        
        # If path is not absolute, then use it relative to the modules root path.
        if not os.path.isabs(module_path):
            module_path = os.path.join(modules_root, module_path)

        logger(5, f'\tadd path {module_path} to path')

        sys.path.append(module_path)
