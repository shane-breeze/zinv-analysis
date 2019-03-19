import awkward as awk
import numpy as np
np.warnings.filterwarnings('ignore')

from AwkwardOps import get_nth_object, jagged_prod

class Lambda(object):
    def __init__(self, function):
        self.function = function

    def begin(self):
        self.lambda_function = eval('lambda '+self.function)

    def __call__(self, *args, **kwargs):
        """
        Pass args and kwargs to the lambda function constructed from a string

        Parameters
        ----------
        "args" : argument list
        "kwargs" : keyword-argument list

        Returns
        -------
        Output of the function `self.lambda_function`

        Notes
        -----
        Catches exceptions inside the function `self.lambda_function` when it's
        called. These are assumed to be attribute errors, i.e. the input
        args/kwargs don't have the attributes asked for in the string-function.
        """
        if not hasattr(self, "lambda_function"):
            self.begin()
        return self.lambda_function(*args, **kwargs)

    def __add__(self, other):
        sargs, sformula = self.function.split(":")
        oargs, oformula = other.function.split(":")

        if sargs != oargs:
            raise TypeError(
                "Cannot merge arguments: {} and {}".format(sargs, oargs)
            )

        new_function = "{}: ({}) & ({})".format(sargs, sformula, oformula)
        return Lambda(new_function)

    def __mul__(self, other):
        sargs, sformula = self.function.split(":")
        oargs, oformula = other.function.split(":")

        if sargs != oargs:
            raise TypeError(
                "Cannot merge arguments: {} and {}".format(sargs, oargs)
            )

        new_function = "{}: ({}) * ({})".format(sargs, sformula, oformula)
        return Lambda(new_function)
