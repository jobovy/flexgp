import scipy as sc
import scipy.linalg as linalg

def stable_cholesky(x,tiny):
    """
    NAME:
       stable_cholesky
    PURPOSE:
       Stable version of the cholesky decomposition
    INPUT:
       x - (sc.array) positive definite matrix
       tiny - (double) tiny number to add to the covariance matrix to make the decomposition stable
    OUTPUT:
       L - (matrix) lower matrix
    REVISION HISTORY:
       2009-09-25 - Written - Bovy (NYU)
    """
    thisx= x+tiny*sc.eye(x.shape[0])
    L= linalg.cholesky(thisx,lower=True)
    return L

