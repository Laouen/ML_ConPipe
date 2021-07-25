class FunctionModule():

    def __init__(self, function, parameters):
        self.parameters = parameters
        self.function = function

    def run(self, *args, **kwargs):
        return self.function(
            *args,
            **kwargs,
            **self.parameters
        )
