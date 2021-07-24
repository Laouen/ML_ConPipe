import sys
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
        print(f'El módulo {module_name} no existe')
    except AttributeError:
        print(f'El módulo {module_name} no tiene definido a {obj_name}')

def get_function(function_name):
    func = get_module_object(function_name)

    if not hasattr(func, '__call__'):
        raise NotFunctionModuleError(f'{function_name} is not callable (i.e. a function)')

def get_class(class_name):
    class_obj = get_module_object(class_name)

    if not inspect.isclass(class_obj):
        raise NotClassModuleError(f'{class_name} is not a class')

def add_path_to_modules(module_paths, logger):
    for module_path in module_paths:
        logger(5, f'\tadd path {module_path} to path')
        sys.path.append(module_path)
