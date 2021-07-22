import glob
import sys
import importlib
import inspect

from ConPipe.decorators import singleton
from ConPipe.modules import DataSplit, ModelEvaluation, ModelSelection
from ConPipe.exceptions import ModuleNotFoundError, NotFunctionModuleError, NotClassModuleError

import sklearn

@singleton
class ModuleLoader():

    def __init__(self, module_directories=[], installed_modules=[]):
        self.modules = {
            'ConPipe.DataSplit': DataSplit,
            'ConPipe.ModelEvaluation': ModelEvaluation,
            'ConPipe.ModelSelection': ModelSelection,
            'sklearn': sklearn
        }

        # Import modules from python files 
        for module_dir in module_directories:
            sys.path.append(module_dir)
            for file in glob.iglob(module_dir + '**/*.py', recursive=True):
                module = file.raplace('../','').replace('/','.').replace('.py','')
                self.modules[module] = importlib.import_module(module)
        
        # Import installed modules
        for module in installed_modules:
            self.modules[module] = importlib.import_module(module)

    def _check_if_exists(self, module, object_name):
        if module not in self.modules or hasattr(self.modules[module], object_name):
            raise ModuleNotFoundError(f'Object {module}.{object_name} not found in list of imported modules')

    def _check_if_function_exists(self, module, function_name):

        self._check_if_exists(module, function_name)        
            
        func = getattr(self.modules[module], function_name)
        if not hasattr(func, '__call__'):
            raise NotFunctionModuleError(f'The {module}.{function_name} object is not callable (i.e. a function)')

    def _check_if_class_exists(self, module, class_name):

        self._check_if_exists(module, class_name)        
            
        obj = getattr(self.modules[module], class_name)
        if not inspect.isclass(obj):
            raise NotClassModuleError(f'The {module}.{class_name} object is not a class')

    def get_function(self, module, function_name):
        self._check_if_function_exists(module, function_name)
        return getattr(self.modules[module], function_name)

    def get_class(self, module, class_name):
        self._check_if_class_exists(module, class_name)
        return getattr(self.modules[module], class_name)