import scipy
from flexgp.covarianceClass import *
_MAXH= 10.
class covarFunc (covariance):
    """
    covarFunc powerlawSF: covariance function with a power-law
                          structure function

    powerlawSF(x,x')= s^2-A/2 |x-x'|^gamma

    """
    def __init__(self,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize a powerlawSF object
        OPTIONAL KEYWORD INPUTS:
           logA= or A=
           gamma=
        OUTPUT:
        HISTORY:
           2010-06-21 - Written - Bovy (NYU)
           2010-12-07 - sill = SF(\infty) - Bovy
        """
        self._dict= {}
        if kwargs.has_key('logA'):
            self.logA= kwargs['logA']
        elif kwargs.has_key('A'):
            self.logA= scipy.log(kwargs['A'])
        else:
            self.logA= 0.
        self._dict['logA']= self.logA
        if kwargs.has_key('gamma'):
            self.gamma= kwargs['gamma']
        else:
            self.gamma= 0.
        self._dict['gamma']= self.gamma
        #Define shortcuts
        self.A= scipy.exp(self.logA)

    def evaluate(self,x,xp):
        """
        NAME:
           evaluate
        PURPOSE:
           evaluate the power-law SF covariance function
        INPUT:
           x - one point
           xp - another point
        OUTPUT:
           covariance
        HISTORY:
           2010-06-21 - Written - Bovy (NYU)
        """
        if self.gamma > 2. or self.gamma < 0.: return -9999.99
        if not isinstance(x,numpy.ndarray):
            x= numpy.array(x)
        if not isinstance(xp,numpy.ndarray):
            xp= numpy.array(xp)
        return 0.5*(self._sf(_MAXH)-self._sf(x-xp))

    def _sf(self,x):
        if numpy.fabs(x) > _MAXH: return self._sf(_MAXH)
        return self.A*numpy.fabs(x)**self.gamma
    
    def deriv(self,x,xp,key=None,covarValue=None):
        """
        NAME:
           deriv
        PURPOSE:
           derivative of the covariance function wrt a hyper-parameter or
           a pseudo-input if key == None
        INPUT:
           x - one point
           xp - another point
           key - key corresponding to the desired hyper-parameter in _dict or 'pseudo' if key == None
        OPTIONAL INPUT:
           covarValue - value of the covariance for these two points
        OUTPUT:
           derivative
        HISTORY:
           2010-07-05 - Written - Bovy (NYU)
        """
        if not covarValue == None:
            covarValue= self.evaluate(x,xp)
        if key == 'logA':
            return -self.A/2.*numpy.fabs(x-xp)**self.gamma
        elif key == 'gamma':
            if x == xp:
                return 0.
            else:
                return -self.A/2.*numpy.fabs(x-xp)**self.gamma*numpy.log(numpy.fabs(x-xp))
        
    def _list_params(self):
        """
        NAME:
           list_params
        PURPOSE:
           list all of the hyper-parameters of this covariance function
        INPUT:
        OUTPUT:
           (list of hyper-parameters (['a','l']),
           dimensionality of the hyper-parameter
        HISTORY:
           2010-02-15 - Written - Bovy (NYU)
        """
        return self._dict.keys()

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
        out= covariance.isDomainFinite(self)
        out['gamma']= [True,True] #All but gamma are infinite
        return out

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
        out= covariance.paramsDomain(self)
        out['gamma']= [0.,2.] #All but gamma are infinite
        return out

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
        out= covariance.create_method(self)
        out['gamma']= 'whole' #All but gamma are default, stepping out=quick
        return out
