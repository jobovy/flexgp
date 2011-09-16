import scipy
from flexgp.meanClass import *
class meanFunc (mean):
    """zeroMean: mean function for a GP with zero mean"""
    def __init__(self,**kwargs):
        self._dict= {}
        return None

    def evaluate(self,x):
        return 0.

    def list_params(self):
        return []

