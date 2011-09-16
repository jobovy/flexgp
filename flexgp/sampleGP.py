###############################################################################
#   sampleGP: sample from a GP posterior
#
#   main routine: sampleGP
###############################################################################
import inspect
import numpy
import bovy_mcmc
from trainGP import *
def sampleGP(trainingSet,covar,mean=None,nsamples=100,
             step=None,fix=None,metropolis=False,markovpy=False):
    """
    NAME:
       sampleGP
    PURPOSE:
       sample a GP
    INPUT:
       trainingSet - a trainingSet instance
       covar - an instance of your covariance function of
               choice, with initialized parameters
       mean= an instance of your mean function of choice, with initialized
             parameters
       nsamples - number of samples desired
       step= step-size for slice creation or metropolis sampling 
             (number or list)
       fix= None or list of parameters to hold fixed
       metropolis= if True, use Metropolis sampling
       markovpy= if True, use markovpy sampling
    OUTPUT:
       list of outcovarFunc
    HISTORY:
       2010-08-08 - Written - Bovy (NYU)
    """
    #Put in dummy mean if mean is None
    if mean is None:
        from flexgp.zeroMean import meanFunc
        mean= meanFunc()
        noMean= True
        out= [covar]
    else: 
        noMean= False
        out= [(covar,mean)]
    #Pack the covariance parameters
    (params,packing)= pack_params(covar,mean,fix)
    if step is None:
        step= [0.1 for ii in range(len(params))]
    #Grab the covariance class
    covarFuncName= inspect.getmodule(covar).__name__
    thisCovarClass= __import__(covarFuncName)
    meanFuncName= inspect.getmodule(mean).__name__
    thisMeanClass= __import__(meanFuncName)
    #Set up isDomainFinite, domain, and create_method, even when metropolis
    isDomainFinite, domain, create_method= [], [], []
    covarIsDomainFinite= covar.isDomainFinite()
    covarDomain= covar.paramsDomain()
    covarCreate= covar.create_method()
    meanIsDomainFinite= mean.isDomainFinite()
    meanDomain= mean.paramsDomain()
    meanCreate= mean.create_method()
    for ii in range(len(packing.GPhyperPackingList)):
        p= packing.GPhyperPackingList[ii]
        try:
            for jj in range(packing.GPhyperPackingDim[ii]):
                isDomainFinite.append(covarIsDomainFinite[p])
                domain.append(covarDomain[p])
                create_method.append(covarCreate[p])
        except KeyError:
            for jj in range(len(packing.GPhyperPackingDim[ii])):
                isDomainFinite.append(meanIsDomainFinite[p])
                domain.append(meanDomain[p])
                create_method.append(meanCreate[p])
    if len(packing.GPhyperPackingList) == 1: #one-d
        isDomainFinite= isDomainFinite[0]
        domain= domain[0]
        create_method= create_method[0]
        if isinstance(step,(list,numpy.ndarray)): step= step[0]
    if not metropolis and not markovpy:
        #slice sample the marginal likelihood
        samples= bovy_mcmc.slice(params,step,
                                 _lnpdf,(trainingSet,packing,thisCovarClass,
                                         thisMeanClass),
                                 isDomainFinite=isDomainFinite,
                                 domain=domain,
                                 nsamples=nsamples,
                                 create_method=create_method)
    elif metropolis:
        samples, faccept= bovy_mcmc.metropolis(params,step,
                                               _lnpdf,(trainingSet,packing,
                                                       thisCovarClass,
                                                       thisMeanClass),
                                               symmetric=True,
                                               nsamples=nsamples)
        if numpy.any((faccept < 0.15)) or numpy.any((faccept > 0.6)):
            print "WARNING: Metropolis acceptance ratio was < 0.15 or > 0.6 for a direction" 
            print "Full acceptance ratio list:"
            print faccept
    elif markovpy:
        samples= bovy_mcmc.markovpy(params,step,
                                    _lnpdf,(trainingSet,packing,thisCovarClass,
                                            thisMeanClass),
                                    isDomainFinite=isDomainFinite,
                                    domain=domain,
                                    nsamples=nsamples)
        print nsamples, len(samples)
    if noMean:
        for ii in range(nsamples):
            hyperParamsDict= unpack_params(samples[ii],packing)
            out.append(packing.covarFunc(**hyperParamsDict))
        return out
    else:
        for ii in range(nsamples):
            hyperParamsDict= unpack_params(samples[ii],packing)
            out.append((packing.covarFunc(**hyperParamsDict),
                        packing.meanFunc(**hyperParamsDict)))
        return out
    
def _lnpdf(params,trainingSet,PackParams,thisCovarClass,thisMeanClass):
    return -marginalLikelihood(params,trainingSet,PackParams,thisCovarClass,
                               thisMeanClass)
