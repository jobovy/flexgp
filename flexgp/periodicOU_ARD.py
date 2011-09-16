import scipy
from flexgp.covarianceClass import *
from flexgp.fast_cholesky import fast_cholesky_invert
class covarFunc (covariance):
    """
    covarFunc periodicOU_ARD: Ornstein-Uhlenbeck covariance function with 
    Automatic Relevance Determination and a period

    periodicOU_ARD(x,x') = a^2 exp[ -\sum_{d=1}^D |sin(pi/P_d*(x_d-x'_d))|/l_d ]
    
    """
    def __init__(self,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize a periodicOU_ARD object
        OPTIONAL KEYWORD INPUTS:
           loga2= or a=
           logl= or l= double, or list/array of doubles; length scales
           logP= or P= double, or list/array of doubles; period
           dim= dimension
        OUTPUT:
        HISTORY:
           2010-06-27 - Written (OU_ARD) - Bovy (NYU)
           2011-06-14 - Written - Bovy (NYU)
        """
        self._dict= {}
        if kwargs.has_key('loga2'):
            self.loga2= kwargs['loga2']
        elif kwargs.has_key('a'):
            self.loga2= 2.*scipy.log(kwargs['a'])
        else:
            self.loga2= 0.
        self._dict['loga2']= self.loga2
        if kwargs.has_key('dim'):
            self.dim= kwargs['dim']
        if kwargs.has_key('logl'):
            (self.logl,tmpdim)= self._process_l_input(kwargs['logl'])
        elif kwargs.has_key('l'):
            (tmpl,tmpdim)= self._process_l_input(kwargs['l'])
            self.logl= scipy.log(tmpl)
        else:
            self.logl= numpy.array([0.])
            tmpdim= 1
        self._dict['logl']= self.logl
        if kwargs.has_key('logP'):
            (self.logP,tmpdim)= self._process_l_input(kwargs['logP'])
        elif kwargs.has_key('P'):
            (tmpP,tmpdim)= self._process_l_input(kwargs['P'])
            self.logP= scipy.log(tmpP)
        else:
            self.logP= numpy.array([0.])
            tmpdim= 1
        self._dict['logP']= self.logP
        if not tmpdim == None:
            self.dim= tmpdim
        if not hasattr(self,'dim'):
            self.dim= len(self.logl)
        #Define shortcuts
        self.a2= scipy.exp(self.loga2)
        self.l= scipy.exp(self.logl)
        self.P= scipy.exp(self.logP)

    def _process_l_input(self,l_in):
        """Internal function to process the l input"""
        outdim= None
        if isinstance(l_in,float):
            if hasattr(self,'dim'):
                out= numpy.ones(self.dim)*l_in
            else:
                outdim= 1
                out= numpy.array([l_in])
        elif isinstance(l_in,list):
            out= numpy.array(l_in,dtype=numpy.float64)
        elif isinstance(l_in,numpy.ndarray):
            out= l_in
        else:
            try:
                l_in[0]
            except TypeError:
                try:
                    tmpl= float(l_in)
                except ValueError:
                    raise covarianceError("'logl= or l=' length parameter should be a float, list of floats, or numpy array")
                else:
                    if hasattr(self,'dim'):
                        out= numpy.ones(self.dim)*tmpl
                    else:
                        outdim= 1
                        out= numpy.array([tmpl])
            else:
                out= numpy.array(l_in,dtype=numpy.float64)
        return (out, outdim)

    def evaluate(self,x,xp):
        """
        NAME:
           evaluate
        PURPOSE:
           evaluate the Ornstein-Uhlenbeck covariance function
        INPUT:
           x - one point
           xp - another point
        OUTPUT:
           covariance
        HISTORY:
           2010-06-27 - Written (OU_ARD) - Bovy (NYU)
           2010-06-14 - Adapted for period - Bovy (NYU)
        """
        if not isinstance(x,numpy.ndarray):
            x= numpy.array(x)
        if not isinstance(xp,numpy.ndarray):
            xp= numpy.array(xp)
        return self.a2*numpy.exp(-numpy.sum(numpy.fabs(numpy.sin(\
                        numpy.pi/self.P*(x-xp)))/self.l))

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

