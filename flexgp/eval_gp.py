###############################################################################
#   eval_gp: module for evaluating a GP
#
#   main routine: eval_gp
#
#   example usage:
#
#      eval_gp(xs,meanfunc,covarfunc,params_mean,params_covar,
#              constraints=[],nGP=1)
#
#   where xs is a list of abcissae, meanfunc and covarfunc are functions that
#   take an 'x' and params_mean (or params_covar; lists, not *params) 
#   and return the mean or covariance or they are instances of meanClass or 
#   CovarianceClass classes, 
#   constraints is an numpy.array(N,3) object of x,y,noise_y 
#   constraints or a trainingSet instance, nGP is the number of samples you 
#   want
#   tiny_cholesky is a small number that can be added to the diagonal to 
#   stabelize the Cholesky decomposition
###############################################################################
import scipy as sc
import scipy.stats as stats
import scipy.linalg as linalg
from stable_cholesky import stable_cholesky
from trainingSet import trainingSet
import meanClass
import covarianceClass
def mean_func(x,params):
    return params.evaluate(x)
def covar_func(x,y,params):
    """covar: covariance function"""
    return params.evaluate(x,y)
def eval_gp(xs,mean,covar,params_mean,params_covar,tiny_cholesky=.0001,
            constraints=[],nGP=1):
    """
    NAME:
       eval_gp
    PURPOSE:
       evaluate nGP function samples from a Gaussian process
    INPUT:
       xs - x-values for which to evaluate the GP
       mean - (function) mean function or meanClass instance
       covar - (function) covariance function or covarianceClass instance
       params_mean - (tuple) parameters given to the mean function
       params_covar - (tuple) parameters given to the covar function
       tiny_cholesky - (double) tiny number to make the cholesky decomposition stable
       constraints - sc.array(N,3) or trainingSetObject
       nGP - (int) number of samples from the GP to plot
    OUTPUT:
       GPsamples - sc.array(nGP,nx) - array with samples for the various GPs
    REVISION HISTORY:
       2009-09-25 - Written - Bovy (NYU)
    """
    if isinstance(mean,meanClass.mean):
        params_mean= mean
        mean= mean_func
    if isinstance(covar,covarianceClass.covariance):
        params_covar= covar
        covar= covar_func
    nx= len(xs)
    means= calc_constrained_mean(xs,mean,params_mean,covar,params_covar,constraints)
    covars= calc_constrained_covar(xs,covar,params_covar,constraints)
    #Compute the cholesky_decomp
    Lcholesky= stable_cholesky(covars,tiny_cholesky)
    GPsamples= sc.zeros((nGP,nx))
    for ii in range(nGP):
        #Generate a vector in which the elements ~N(0,1)
        tmp_ys= stats.norm.rvs(size=nx)
        #Form the sample as Ly+mean
        GPsamples[ii,:]= sc.dot(Lcholesky,tmp_ys)+means
    return GPsamples

def calc_constrained_mean(xs,mean,params_mean,covar,params_covar,constraints):
    """
    NAME:
       calc_constrained_mean
    PURPOSE:
       calculate the mean of a GP when there are constraints
    INPUT:
       xs - (array of doubles) evaluate the function at xs
       mean - mean function
       params_mean - parameters for the original mean function
       covar - original covariance function
       params_covar - parameters of the covariance function
       constraints - list of constraints ([x,y,s_y])
    OUTPUT:
       constrained mean function evaluated at x
    REVISION HISTORY:
       2009-09-25 - Written - Bovy (NYU)
    """
    if isinstance(mean,meanClass.mean):
        params_mean= mean
        mean= mean_func
    if isinstance(covar,covarianceClass.covariance):
        params_covar= covar
        covar= covar_func
    nx= len(xs)
    unconstrained_means= sc.zeros(nx)
    for ii in range(nx):
        unconstrained_means[ii]= mean(xs[ii],params_mean)
    if constraints is None:
        return unconstrained_means
    if isinstance(constraints,trainingSet):
        nconstraints= len(constraints.listy)
    else:
        nconstraints= constraints.shape[0]
    if nconstraints == 0:
        return unconstrained_means
    else:
        constraints_means= sc.zeros(nconstraints)
        for ii in range(nconstraints):
            if isinstance(constraints,trainingSet):
                constraints_means[ii]= mean(constraints.listx[ii],params_mean)
            else:
                constraints_means[ii]= mean(constraints[ii,0],params_mean)
        if isinstance(constraints,trainingSet):
            diffymx= constraints.listy-constraints_means
        else:
            diffymx= constraints[:,1]-constraints_means
        constraints_covars= sc.zeros((nconstraints,nconstraints))
        for ii in range(nconstraints):
            for jj in range(ii,nconstraints):
                if isinstance(constraints,trainingSet):
                    constraints_covars[ii,jj]=covar(constraints.listx[ii],constraints.listx[jj],params_covar)+( ii == jj )*constraints.noiseCovar[ii]
                else:
                    constraints_covars[ii,jj]=covar(constraints.listx[ii],constraints.listx[jj],params_covar)+( ii == jj )*constraints.noiseCovar[ii]
                constraints_covars[jj,ii]= constraints_covars[ii,jj]
        proj_covars= sc.zeros((nx,nconstraints))
        for ii in range(nconstraints):
            for jj in range(nx):
                if isinstance(constraints,trainingSet):
                    proj_covars[jj,ii]= covar(xs[jj],constraints.listx[ii],params_covar)
                else:
                    proj_covars[jj,ii]= covar(xs[jj],constraints[ii,0],params_covar)
        return unconstrained_means+sc.dot(sc.dot(proj_covars,linalg.inv(constraints_covars)),diffymx)

def calc_constrained_covar(xs,covar,params_covar,constraints):
    """
    NAME:
       calc_constrained_covar
    PURPOSE:
       calculate the covariance of a GP if there are constraints
    INPUT:
       xs - x values to calculate the covariance of
       covar - (function) the original covariance function
       params_covar - parameters to be passed to the original covariance function
       constraints - list of constraints ([x,y,s_y])
    OUTPUT:
       constrained covariance evaluated at (x,y)
    REVISION HISTORY:
       2009-09-25 - Written - Bovy (NYU)
    """
    if isinstance(covar,covarianceClass.covariance):
        params_covar= covar
        covar= covar_func
    nx= len(xs)
    unconstrained_covars= sc.zeros((nx,nx))
    for ii in range(nx):
        for jj in range(ii,nx):
            unconstrained_covars[ii,jj]= covar(xs[ii],xs[jj],params_covar)
            unconstrained_covars[jj,ii]= unconstrained_covars[ii,jj]
    if constraints is None:
        return unconstrained_covars
    if isinstance(constraints,trainingSet):
        nconstraints= len(constraints.listy)
    else:
        nconstraints= constraints.shape[0]
    if nconstraints == 0:
        return unconstrained_covars
    else:
        constraints_covars= sc.zeros((nconstraints,nconstraints))
        for ii in range(nconstraints):
            for jj in range(ii,nconstraints):
                if isinstance(constraints,trainingSet):
                    constraints_covars[ii,jj]=covar(constraints.listx[ii],constraints.listx[jj],params_covar)+( ii == jj )*constraints.noiseCovar[ii]
                else:
                    constraints_covars[ii,jj]=covar(constraints[ii,0],constraints[jj,0],params_covar)+( ii == jj )*constraints[ii,2]**2.
                constraints_covars[jj,ii]= constraints_covars[ii,jj]
        proj_covars= sc.zeros((nx,nconstraints))
        for ii in range(nconstraints):
            for jj in range(nx):
                if isinstance(constraints,trainingSet):
                    proj_covars[jj,ii]= covar(xs[jj],constraints.listx[ii],params_covar)
                else:
                    proj_covars[jj,ii]= covar(xs[jj],constraints[ii,0],params_covar)
        return unconstrained_covars - sc.dot(sc.dot(proj_covars,linalg.inv(constraints_covars)),proj_covars.T)

