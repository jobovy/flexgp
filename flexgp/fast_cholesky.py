import numpy
import scipy as sc
import scipy.linalg as linalg
_TINY= 0.000000001
def stable_cho_factor(x,tiny=_TINY):
    """
    NAME:
       stable_cho_factor
    PURPOSE:
       Stable version of the cholesky decomposition
    INPUT:
       x - (sc.array) positive definite matrix
       tiny - (double) tiny number to add to the covariance matrix to make the decomposition stable (has a default)
    OUTPUT:
       (L,lowerFlag) - output from scipy.linalg.cho_factor for lower=True
    REVISION HISTORY:
       2009-09-25 - Written - Bovy (NYU)
    """
    
    return linalg.cho_factor(x+sc.sum(sc.diag(x))*tiny*sc.eye(x.shape[0]),lower=True)
    
def fast_cholesky(A,logdet=False,tiny=_TINY):
    """
    NAME:
       fast_cholesky
    PURPOSE:
       Calculate the Choleksy decomposition of a positive definite matrix
       Adds some noise to stabilize the decomposition
    INPUT:
       A - matrix to be Cholesky decomposed
       logdet - (Bool) if True, return the logarithm of the determinant as well
       tiny - (double) tiny number to add to the covariance matrix to make the decomposition stable (has a default)
    OUTPUT:
       (L,lowerFlag) - output from scipy.linalg.cho_factor for lower=True
       + logdet if asked for
    REVISION HISTORY:
       2009-10-07 - Written - Bovy (NYU)
    """
    L= stable_cho_factor(A,tiny=tiny)
    if logdet:
        return (L,2.*sc.sum(sc.log(sc.diag(L[0]))))
    else:
        return L
    
def fast_cholesky_invert(A,logdet=False,tiny=_TINY):
    """
    NAME:
       fast_cholesky_invert
    PURPOSE:
       invert a positive definite matrix by using its Cholesky decomposition
    INPUT:
       A - matrix to be inverted
       logdet - (Bool) if True, return the logarithm of the determinant as well
       tiny - (double) tiny number to add to the covariance matrix to make the decomposition stable (has a default)
    OUTPUT:
       A^{-1}
    REVISION HISTORY:
       2009-10-07 - Written - Bovy (NYU)
    """
    L= stable_cho_factor(A,tiny=tiny)
    if logdet:
        return (linalg.cho_solve(L,sc.eye(A.shape[0])),2.*sc.sum(sc.log(sc.diag(L[0]))))
    else:
        return linalg.cho_solve(L,sc.eye(A.shape[0]))
