from ConPipe.decorators import singleton

@singleton
class Logger():
    def __init__(self, verbose=1):
        self.verbose = verbose

    def log(self, verbose, message, ident=0):
        if verbose <= self.verbose:
            print(''.join(['\t' for _ in range(ident)]) + message)
    
    def __call__(self, verbose, message, ident=0):
        self.log(verbose, message, ident)
