import scipy
from gp.meanClass import *
class meanFunc (mean):
    """constoMean: mean function for a GP with zero mean"""
    def __init__(self,m=0.,**kwargs):
        self.m= m
        self._dict= {}
        self._dict['m']= m

    def evaluate(self,x):
        return self.m

    def list_params(self):
        return self._dict.keys()

