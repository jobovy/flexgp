###############################################################################
#   trainGP: train a GP
#
#   main routine: trainGP
#
#   example usage:
#
#      trainGP(trainingSetObject,covarObject,useDerivs=False,mean=meanObject)
#
#   where trainingSetObject, covarObject, and meanObjects are instances of the
#   trainingSet, covarClass, and meanClass classes
###############################################################################
import inspect
import numpy
import scipy
import scipy.optimize as optimize
import scipy.misc
from gp.fast_cholesky import fast_cholesky_invert
from gp.covarianceClass import covarianceError
_LOGTWOPI= numpy.log(2.*numpy.pi)
def trainGP(trainingSet,covar,mean=None,filename=None,ext='h5',
            useDerivs=True,fix=None):
    """
    NAME:
       trainGP
    PURPOSE:
       train a GP
    INPUT:
       trainingSet - a trainingSet instance
       covar - an instance of your covariance function of
               choice, with initialized parameters
       mean= an instance of your mean function of choice, with initialized
             parameters
       filename - filename for output
       ext - format of the output file (checks extension if none given,
             if this is not 'fit' or 'fits' assumes HDF5     
       useDerivs - use the derivative of the objective function to optimize
       fix= fix these parameters of the mean and covariance functions
    OUTPUT:
       trained GP= outcovarFunc
    HISTORY:
       2010-07-25 - Written - Bovy (NYU)
    """
    #Put in dummy mean if mean is None
    if mean is None:
        from gp.zeroMean import meanFunc
        mean= meanFunc()
        noMean= True
    else: noMean= False
    #Pack the covariance and mean parameters
    (params,packing)= pack_params(covar,mean,fix)
    #Grab the covariance and mean class
    covarFuncName= inspect.getmodule(covar).__name__
    thisCovarClass= __import__(covarFuncName)
    meanFuncName= inspect.getmodule(mean).__name__
    thisMeanClass= __import__(meanFuncName)
    #Optimize the marginal likelihood
    #Sort of good ftol, assuming data has been pre-processed
    try:
        if useDerivs:
            raise NotImplementedError()
            params= optimize.fmin_cg(marginalLikelihood,params,
                                     fprime=derivMarginalLikelihood,
                                     args=(trainingSet,packing,
                                           thisCovarClass,useDerivs))
        else:
            params= optimize.fmin_powell(marginalLikelihood,params,
                                         args=(trainingSet,packing,
                                               thisCovarClass,
                                               thisMeanClass,
                                               useDerivs))
    except numpy.linalg.linalg.LinAlgError:
        raise
    if not filename == None:
        write_train_out(params,packing,filename=filename,ext=ext)
    if packing.nhyperParams == 1: params= numpy.array([params])
    hyperParamsDict= unpack_params(params,packing)
    outcovarFunc= packing.covarFunc(**hyperParamsDict)
    if noMean:
        return outcovarFunc
    outmeanFunc= packing.meanFunc(**hyperParamsDict)
    return (outcovarFunc,outmeanFunc)
    
def print_current_params(params):
    print params

def _calc_ky(covarFunc,trainingSet,N):
    """Internal function to calculate Ky"""
    Ky= scipy.zeros((N,N))
    for ii in range(N):
        Ky[ii,ii]= covarFunc.evaluate(trainingSet.listx[ii],
                                      trainingSet.listx[ii])
        if trainingSet.hasNoise:
            Ky[ii,ii]+= trainingSet.noiseCovar[ii]
        for jj in range(ii+1,N):
            Ky[ii,jj]= covarFunc.evaluate(trainingSet.listx[ii],
                                          trainingSet.listx[jj])
            Ky[jj,ii]= Ky[ii,jj]
    return Ky

def _calc_mean(meanFunc,trainingSet,N):
    """Internal function to calculate the mean"""
    out= numpy.zeros(N)
    for ii in range(N):
        out[ii]= meanFunc.evaluate(trainingSet.listx[ii])
    return out
        
def marginalLikelihood(params,trainingSet,PackParams,thisCovarClass,
                       thisMeanClass=None,
                       useDerivs=False):
    """
    NAME:
       marginalLikelihood
    PURPOSE:
       evaluate minus the marginal (log) likelihood
    INPUT:
       params - ndarray consisting of
          covarHyper - a set of covariance Hyperparameters
       These can be unpacked by using the PackParams
       trainingSet - trainingSet object
       PackParams - parameters describing the GP (to unpack the parameters)
       thisCovarClass - covariance class (not used in this function)
       thisMeanClass - same as previous two, but for the mean 
                                       function
    OUTPUT:
       log of the marginal likelihood
    HISTORY:
       2010-07-25 - Written - Bovy (NYU)
    """
    hyperParamsDict= unpack_params(params,PackParams)
    covarFunc= PackParams.covarFunc(**hyperParamsDict)
    meanFunc= PackParams.meanFunc(**hyperParamsDict)
    N= trainingSet.nTraining
    #Calculate Ky
    Ky= _calc_ky(covarFunc,trainingSet,N)
    try:
        (Kyinv,Kylogdet)= fast_cholesky_invert(Ky,logdet=True)
    except (numpy.linalg.linalg.LinAlgError,ValueError):
        if useDerivs:
            print "Warning: parameter space includes non-positive definite regions; consider using useDerivs=False"
        return numpy.finfo(numpy.dtype(numpy.float64)).max
    datamean= _calc_mean(meanFunc,trainingSet,N)
    chi2= numpy.dot((trainingSet.listy-datamean),
                    numpy.dot(Kyinv,(trainingSet.listy-datamean)))
    return 0.5*(chi2+Kylogdet+N*_LOGTWOPI)
    
def derivMarginalLikelihood(params,trainingSet,PackParams,thisCovarClass,
                            useDerivs):
    """
    NAME:
       derivMarginalLikelihood
    PURPOSE:
       evaluate the derivative of the marginal (log) likelihood
    INPUT:
       params - ndarray consisting of
          covarHyper - a set of covariance Hyperparameters
       These can be unpacked by using the GPparams
       trainingSet - trainingSet object
       PackParams - parameters describing the GP (to unpack the parameters)
       thisCovarClass - covariance class
    OUTPUT:
       derivative of the log of the marginal likelihood
    HISTORY:
       2010-07-25 - Written - Bovy (NYU)
    """
    hyperParamsDict= unpack_params(params,PackParams)
    covarFunc= PackParams.covarFunc(**hyperParamsDict)
    N= trainingSet.nTraining
    #Calculate Ky
    Ky= _calc_ky(covarFunc,trainingSet,N)
    try:
        (Kyinv,Kylogdet)= fast_cholesky_invert(Ky,logdet=True)
    except (numpy.linalg.linalg.LinAlgError,ValueError):
        print "Warning: "
        return -numpy.finfo(numpy.dtype(numpy.float64)).max
    #All the derivatives return dictionaries with keys corresponding to the 
    #hyper-parameters
    Kydothyper= _calc_kydot_hyper(covarFunc,trainingSet,N,hyperParamsDict)
    alpha= numpy.dot(Kyinv,trainingSet.listy)
    aat= numpy.outer(alpha,alpha)
    Ldothyper= {}
    for key in hyperParamsDict:
        Ldothyper[key]= 0.5*numpy.trace(numpy.dot(Kyinv-aat,Kydothyper[key]))
    #Now pack the derivatives, by cleverly re-using the packing routine
    fakeCovar= thisCovarClass.covarFunc(**Ldothyper)
    (derivs,derivspacking)= pack_params(fakeCovar,None,None)
    #print "derivs", derivs
    return derivs

def _calc_kydot_hyper(covarFunc,trainingSet,N,hyperParamsDict):
    """Internal function to calculate Kydot for covariance function hyperparameters"""
    out= {}
    for key in hyperParamsDict:
        thisOut= scipy.zeros((N,N))
        for ii in range(N):
            thisOut[ii,ii]= covarFunc.deriv(trainingSet.listx[ii],
                                            trainingSet.listx[ii],
                                            key)
            for jj in range(ii+1,N):
                thisOut[ii,jj]= covarFunc.deriv(trainingSet.listx[ii],
                                                trainingSet.listx[jj],
                                                key)
                thisOut[jj,ii]= thisOut[ii,jj]
        out[key]= thisOut
    return out

def pack_params(covar,mean,fix):
    """
    NAME:
       pack_params
    PURPOSE:
       pack the hyperparameters and pseudo-inputs into an ndarray
       for the optimization routine(s)
    INPUT:
       covar - covariance function that hyperParams are the
               hyper parameters of
       mean - mean function that you also want to pack the parameters of
       fix - None or array of parameters to fix
    OUTPUT:
       (params,PackParams): ndarray and object that describes the packing
    HISTORY:
       2010-02-12 - Written - Bovy (NYU)
       2010-07-25 - Adapted for trainGP - Bovy
       2011-07-04 - Added fix - Bovy
    """
    PackParams= GPPackParams()
    PackParams.covarFunc= covar.__class__
    PackParams.meanFunc= mean.__class__
    GPhyperPackingList= covar.list_params()
    GPhyperPackingList.extend(mean.list_params())
    GPhyperPackingList= list(set(GPhyperPackingList))
    #Grab those parameters that need to remain fixed
    fixDict= {}
    if not fix is None:
        if isinstance(fix,str): fix= [fix]
        for p in fix:
            if p in GPhyperPackingList:
                try:
                    fixDict[p]= covar[p]
                except KeyError:
                    fixDict[p]= mean[p]
                GPhyperPackingList.remove(p)
    PackParams.fixDict= fixDict
    if len(GPhyperPackingList) == 0:
        PackParams.nhyperParams= 0
        return ([],PackParams)
    GPhyperPackingDim= _get_hyper_dims(GPhyperPackingList,covar,mean)
    PackParams.GPhyperPackingList= GPhyperPackingList
    PackParams.GPhyperPackingDim= GPhyperPackingDim
    nhyperParams= scipy.sum(GPhyperPackingDim)
    PackParams.nhyperParams= nhyperParams
    nparams= nhyperParams
    params= scipy.zeros(nparams)
    for ii in range(len(GPhyperPackingDim)):
        #BOVY: Replace these indices with cumulative sum arrays
        try:
            params[scipy.sum(GPhyperPackingDim[0:ii]):scipy.sum(GPhyperPackingDim[0:ii])+GPhyperPackingDim[ii]]= scipy.array(covar[GPhyperPackingList[ii]])
        except KeyError:
            params[scipy.sum(GPhyperPackingDim[0:ii]):scipy.sum(GPhyperPackingDim[0:ii])+GPhyperPackingDim[ii]]= scipy.array(mean[GPhyperPackingList[ii]])
    return (params,PackParams)

def _get_hyper_dims(hyperList,covar,mean):
    """Internal function that finds the dimensionalities of the various hyper-parameters"""
    out= []
    for param in hyperList:
        try:
            thisValue= covar[param]
        except KeyError:
            try:
                thisValue= mean[param]
            except KeyError:
                continue
        if isinstance(thisValue,int):
            out.append(1)
        elif isinstance(thisValue,float):
            out.append(1)
        elif isinstance(thisValue,numpy.float64):
            out.append(1)
        elif isinstance(thisValue,list):
            out.append(len(thisValue))
        elif isinstance(thisValue,numpy.ndarray):
            out.append(len(thisValue))
        else:
            raise covarianceError("I don't understand the format of your "+str(param)+" hyper-parameter; Parameters should be ints, floats, float64s, lists, or one-dimensional ndarrays")
    return out
    
def unpack_params(params,PackParams):
    """
    NAME:
       unpack_params
    PURPOSE:
       unpack the parameters given to the marginal likelihood
       and its derivatives
    INPUT:
       params - ndarray of parameters
       PackParams - parameters describing the packing
    OUTPUT:
       hyperParams object
    HISTORY:
       2010-07-25 - Written - Bovy (NYU)
    """
    hyperParamsDict= {}
    for ii in range(PackParams.nhyperParams):
        hyperParamsDict[PackParams.GPhyperPackingList[ii]]= params[scipy.sum(PackParams.GPhyperPackingDim[0:ii]):scipy.sum(PackParams.GPhyperPackingDim[0:ii])+PackParams.GPhyperPackingDim[ii]]
    return dict(hyperParamsDict, **PackParams.fixDict)

def write_train_out(params,packing,filename='train.out',ext=None):
    """
    NAME:
       write_train_out
    PURPOSE:
       write the output of the training-procedure to a file
    INPUT:
       params
       packing
    OPTIONAL INPUT:
       filename - filename for output file
       ext - format of the output file (checks extension if none given,
             if this is not 'fit' or 'fits' assumes HDF5
    OUTPUT:
    HISTORY:
       2010-02-17 - Written - Bovy (NYU)
    """
    if ext == None:
        import re
        tmp_ext= re.split('\.',filename)[-1]
        if tmp_ext == 'fit' or tmp_ext == 'fits':
            ext= 'fit'
    if ext == 'h5' or ext == None:
        _write_train_out_h5(params,packing,filename)

def read_train_out(filename='train.out',ext=None):
    """
    NAME:
       read_train_out
    PURPOSE:
       read a file containing the output of a training-procedure
    OPTIONAL INPUT:
       filename - name of the file holding the training-output
       ext - format of the output file (checks extension if none given,
             if this is not 'fit' or 'fits' assumes HDF5       
    OUTPUT:
       outcovarFunc - covariance instance
    HISTORY:
       2010-02-17 - Written - Bovy (NYU)
    """
    if ext == None:
        import re
        tmp_ext= re.split('\.',filename)[-1]
        if tmp_ext == 'fit' or tmp_ext == 'fits':
            ext= 'fit'
    if ext == 'h5' or ext == None:
        return _read_train_out_h5(filename)

def _read_train_out_h5(filename):
    """Internal function that reads the training-output from a hdf5 file"""
    import tables
    infile = tables.openFile(filename,mode = "r")
    outcovarFunc = _read_covar_h5(infile)
    infile.close()
    return outcovarFunc
    
def _write_train_out_h5(params,packing,filename):
    """Internal function that writes the training-output to an hdf5 file"""
    import tables
    hyperParamsDict= unpack_params(params,packing)
    outfile = tables.openFile(filename,mode = "w")
    _write_covar_h5(outfile,hyperParamsDict,packing)
    outfile.close()        

def _read_covar_h5(infile):
    #Covariance
    covarFuncName= scipy.array(infile.root.covarGroup.covarClass)[0]
    covarDict= {}
    for node in infile.listNodes(infile.root.covarGroup):
        if node._v_name == 'covarClass':
            continue
        thisValue= scipy.array(node)
        if len(thisValue.flatten()) == 1:
            thisValue= thisValue[0]
        covarDict[node._v_name]= thisValue
    thisCovarClass= __import__(covarFuncName)
    outcovarFunc= thisCovarClass.covarFunc(**covarDict)
    return outcovarFunc

def _write_covar_h5(outfile,hyperParamsDict,packing):
    """Internal function that takes care of part of the writing to file"""
    covarGroup= outfile.createGroup("/",'covarGroup','Covariance function')
    #Covariance function params
    tmpcovar= packing.covarFunc(**hyperParamsDict)#To get the name of module
    outfile.createArray(covarGroup,'covarClass',[inspect.getmodule(tmpcovar).__name__])
    for key, value in hyperParamsDict.iteritems():
        if isinstance(value,numpy.ndarray):
            outfile.createArray(covarGroup, str(key), value)
        elif isinstance(value,float) or isinstance(value,numpy.float64):
            outfile.createArray(covarGroup, str(key), numpy.array([value]))
    return None
    
class GPPackParams:
    """
    Class that describes the GP we are using and the way it
    packs hyper-parameters and pseudo-inputs into an ndarray

    Empty because it is merely a container for a few items
    """
    pass

    def __copy__(self):
        result= GPPackParams()
        members= inspect.getmembers(self)
        for member in members:
            if not inspect.ismethod(member[1]):
                setattr(result,member[0],member[1])
        return result
    
    def __deepcopy__(self, memo={}):
        from copy import deepcopy
        result= GPPackParams()
        members= inspect.getmembers(self)
        for member in members:
            if not inspect.ismethod(member[1]):
                setattr(result,member[0],deepcopy(member[1],memo))
        memo[id(self)] = result
        return result

