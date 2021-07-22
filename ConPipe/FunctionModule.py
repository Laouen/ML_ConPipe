class FunctionModule():

    def __init__(self, function, parameters):
        self.parameters = parameters
        self.function = function

    def run(self, inputs):
        return self.function(
            inputs,
            **self.parameters
        )
