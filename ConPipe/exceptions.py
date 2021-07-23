class NotExistentMethodError(ValueError, AttributeError):
    """ Exception to raise if a class has not method """

class NotFunctionModuleError(ValueError, AttributeError):
    """ Exception to raise if a module is not callable (i.e. can be treated as a function) """

class NotClassModuleError(ValueError, AttributeError):
    """ Exception to raise if a module is not a class """


