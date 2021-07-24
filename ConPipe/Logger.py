from ConPipe.decorators import singleton

@singleton
class Logger():
    def __init__(self, verbose=1):
        self.verbose = verbose

    def log(self, verbose, message):
        if verbose <= self.verbose:
            print(message)
    
    def __call__(self, verbose, message):
        self.log(verbose, message)
