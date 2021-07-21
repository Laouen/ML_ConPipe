from ConPipe.utils import find_function_from_modules

# TODO: include user modules
function_modules = []


def get_function(funct_name):
    return find_function_from_modules(
        function_modules,
        funct_name
    )


class DataSplit():

    def __init__(self, config, verbose):
        self.config = config
        self.verbose = verbose

        self.function = get_function(
            self.config['function']
        )

    def run(self, inputs):

        return self.function(
            inputs,
            **self.config['parameters']
        )
