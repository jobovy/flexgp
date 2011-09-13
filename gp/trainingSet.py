###############################################################################
#   trainingSet: training-set class for training a GP
#
#   main routine: __init__
#
#   example usage:
#
#      trainingSetObject= trainingSet(listx=,listy=,noise=)
#
#   where listx is a list of abcissae, listy is the corresponding set ordinates
#   and noise is the noise in listy
#
#   actual example: trainSet= trainingSet(listx=mjd[band],
#                                  listy=m[band]-numpy.mean(m[band]),
#                                  noise=err_m[band])
###############################################################################
import numpy
class trainingSet:
    """
    trainingSet: Class representing a set of training
    points to train the (SP)GP
    """
    def __init__(self,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           Initialize a training set object
        INPUT:
           Either:
              listx=a list of ndarrays of training set inputs [N,dim]
              listy=a list or ndarray of training set outputs [N]
              noise= list of noise in y
              or noiseCovar= list of noise**2. in y
        OUTPUT:
        HISTORY:
           2010-02-12 - Written - Bovy (NYU)
        """
        if kwargs.has_key('listx'):
            if not isinstance(kwargs['listx'],list) and not isinstance(kwargs['listx'],numpy.ndarray):
                raise trainingSetError("Your 'listx=' object is not a list or ndarray")
            elif isinstance(kwargs['listx'],list):
                self.listx= kwargs['listx']
            else: #is ndarray
                if len(kwargs['listx'].shape) == 1: #one-d
                    self.listx= list(kwargs['listx'])
                else: #more-than-one-d
                    self.listx= [numpy.array(kwargs['listx'][ii,:]) for ii in range(kwargs['listx'].shape[0])]
        if kwargs.has_key('listy'):
            if not isinstance(kwargs['listy'],list) and not isinstance(kwargs['listy'],numpy.ndarray):
                raise trainingSetError("Your 'listy=' object is not a list or ndarray")
            elif isinstance(kwargs['listy'],list):
                self.listy= numpy.array(kwargs['listy'])
            else: #is ndarray
                self.listy= kwargs['listy']
        self.nTraining= len(self.listy)
        if kwargs.has_key('noiseCovar'):
            if isinstance(kwargs['noiseCovar'],float):
                self.noiseCovar= kwargs['noiseCovar']
                self.uniformNoise= True
            elif isinstance(kwargs['noiseCovar'],list):
                self.noiseCovar= numpy.array(kwargs['noiseCovar'],dtype=numpy.float64)
                self.uniformNoise= False
            elif isinstance(kwargs['noiseCovar'],numpy.ndarray):
                self.noiseCovar= kwargs['noiseCovar']
                self.uniformNoise= True
            else:
                try:
                    kwargs['noiseCovar'][0]
                except TypeError:
                    try:
                        tmpnoise= float(kwargs['noiseCovar'])
                    except ValueError:
                        raise trainingSetError("'noiseCovar=' noise parameter should be a float, list of floats, or numpy array")
                    else:
                        self.noiseCovar= tmpnoise
                        self.uniformNoise= True
                else:
                    self.noiseCovar= numpy.array(kwargs['noiseCovar'],dtype=numpy.float64)
                    self.uniformNoise= False
            self.hasNoise= True
        elif kwargs.has_key('noise'):
            if isinstance(kwargs['noise'],float):
                self.noise= kwargs['noise']
                self.uniformNoise= True
            elif isinstance(kwargs['noise'],list):
                self.noise= numpy.array(kwargs['noise'],dtype=numpy.float64)
                self.uniformNoise= False
            elif isinstance(kwargs['noise'],numpy.ndarray):
                self.noise= kwargs['noise']
                self.uniformNoise= True
            else:
                try:
                    kwargs['noise'][0]
                except TypeError:
                    try:
                        tmpnoise= float(kwargs['noise'])
                    except ValueError:
                        raise trainingSetError("'noise=' noise parameter should be a float, list of floats, or numpy array")
                    else:
                        self.noise= tmpnoise
                        self.uniformNoise= True
                else:
                    self.noise= numpy.array(kwargs['noise'],dtype=numpy.float64)
                    self.uniformNoise= False
            self.hasNoise= True
            self.noiseCovar= self.noise**2.
        else:
            self.hasNoise= False

    def __copy__(self):
        if self.hasNoise:
            return self.__class__(listx=self.listx,listy=self.listy,
                                  noiseCovar=self.noiseCovar)
        else:
            return self.__class__(listx=self.listx,listy=self.listy)

    def __deepcopy__(self, memo={}):
        from copy import deepcopy
        newlistx= deepcopy(self.listx,memo)
        newlisty= deepcopy(self.listy,memo)
        if self.hasNoise:
            newnoiseCovar= deepcopy(self.noiseCovar,memo)
        if self.hasNoise:
            return self.__class__(listx=newlistx,listy=newlisty,
                                  noiseCovar=newnoiseCovar)
        else:
            return self.__class__(listx=newlistx,listy=newlisty)
        memo[id(self)] = result
        return result
            
class trainingSetError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


