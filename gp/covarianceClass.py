import numpy
#######################################################################
# covariance API:
#
# 1) All covariance functions must inherit from this covariance class 
#    as covarFunc (covariance)
# 2) All covariance functions must store their hyper-parameters in
#    a '_dict' attribute (a dictionary)
# 3) All covariance functions must implement the evaluate function:
#    evaluate(self,x1,x2)
# 4) covariance functions may implement the deriv function:
#    deriv(self,x1,x2,key=)
#    this function evaluates the derivative, with respect to
#       a) key=hyperparameter_from_dict, or
#       b) key=None: derivative with respect to x1 (not necessary for 
#          regular GPs
# 5) __init__ should always take **kwargs (irrelevant kwargs will be 
#        passed to the __init__ function
# 6) BOVY: something about finite domain for slice sampling here
#######################################################################
class covariance:
    """
    covariance: Top-level class that represents a covariance
    function Specific covariance functions need to inherit from
    this class and implement the placeholder functions defined
    here
    """
    def __init__(self):
        return None
            
    def __getitem__(self, key):
        """
        NAME:
           __getitem__
        PURPOSE:
           get the value of a parameter
        INPUT:
           key - dict key
        OUTPUT:
           parameter value
        HISTORY:
           2010-02-01 - Written - Bovy (NYU)
        """
        try:
            return self._dict[key]
        except AttributeError:
            raise covarianceError("'__getitem__' failed; you must store your hyperparameters in a _dict dictionary")
 
    def __copy__(self):
        return self.__class__(**self._dict)

    def __deepcopy__(self, memo={}):
        from copy import deepcopy
        newdict= deepcopy(self._dict,memo)
        result = self.__class__(**newdict)
        memo[id(self)] = result
        return result

    def evaluate(self,x,xp):
        """
        NAME:
           evaluate
        PURPOSE:
           evaluate the covariance function
        INPUT:
           the two points between which the covariance should be calculated
        OUTPUT:
           output from the covariance(x,xp) function
        HISTORY:
           2010-02-12 - Written - Bovy (NYU)
        """
        raise covarianceError("'evaluate' function of this covariance function is not implemented")

    def list_params(self):
        """
        NAME:
           list_params
        PURPOSE:
           list all of the hyper-parameters of this covariance function
        INPUT:
        OUTPUT:
           list of hyper-parameters
        HISTORY:
           2010-02-15 - Written - Bovy (NYU)
        """
        try:
            return self._dict.keys()
        except AttributeError:
            raise covarianceError("Automatic 'list_params' function failed; define _dict dictionary in your covariance Class or implement your own 'list_params' function")

    def isDomainFinite(self):
        """
        NAME:
           isDomainFinite
        PURPOSE:
           return dictionary that says whether the hyperparameters' domains are finite
        INPUT:
        OUTPUT:
           boolean list
        HISTORY:
           2011-06-13 - Written - Bovy (NYU)
        """
        try:
            out= {}
            for key in self._dict.keys():
                out[key]= [False,False]
            return out
        except AttributeError:
            raise covarianceError("Automatic 'isDomainFinite' function failed; define _dict dictionary in your covariance Class or implement your own 'isDomainFinite' function")

    def paramsDomain(self):
        """
        NAME:
           paramsDomain
        PURPOSE:
           return dictionary that has each hyperparameter's domain 
           (irrelevant for hyperparameters with infinite domains)
        INPUT:
        OUTPUT:
           dictionary of lists
        HISTORY:
           2011-06-13 - Written - Bovy (NYU)
        """
        try:
            out= {}
            for key in self._dict.keys():
                out[key]= [0.,0.]
            return out
        except AttributeError:
            raise covarianceError("Automatic 'paramsDomain' function failed; define _dict dictionary in your covariance Class or implement your own 'paramsDomain' function")

    def create_method(self):
        """
        NAME:
           create_method
        PURPOSE:
           return dictionary that has each hyperparameter's create_method
           for slice sampling
        INPUT:
        OUTPUT:
           dictionary of methods
        HISTORY:
           2011-06-13 - Written - Bovy (NYU)
        """
        try:
            out= {}
            for key in self._dict.keys():
                out[key]= 'step_out'
            return out
        except AttributeError:
            raise covarianceError("Automatic 'create_method' function failed; define _dict dictionary in your covariance Class or implement your own 'create_method' function")

class covarianceError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
