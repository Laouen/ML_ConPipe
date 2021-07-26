from ConPipe.decorators import singleton

@singleton
class Logger():
    def __init__(self, verbose=1):
        self.verbose = verbose

    def log(self, verbose, *msgs, ident=0):
        if verbose <= self.verbose:
            print(''.join(['\t' for _ in range(ident)]), *msgs)
    
    def __call__(self, verbose, *msgs, ident=0):
        self.log(verbose, *msgs, ident)
