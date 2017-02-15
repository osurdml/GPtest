import numpy as np
from scipy.special import ndtr as std_norm_cdf

class NormCDF(object):
    def logpyf(self,y,f):
        return np.log(std_norm_cdf(y*f))

    def dlogpyf_df(self,y,f):
        std_norm_cdf
        y*


def logistic_function(z):
    return 1.0/(1 + np.exp(-z))

def ClassifierLikelihood(object):
    def __init__(self, inverse_link_function=None):
        if inverse_link_function == 'Logit':
            self.inverse_link_function = logistic_function
        else:
            self.inverse_link_function = std_norm_cdf

        
    def _preprocess_values(self):
        """
        Check if the values of the observations correspond to the values
        assumed by the likelihood function.

        ..Note:: Binary classification algorithm works better with classes {-1, 1}
        """
        Y_prep = self.Y.copy()
        Y1 = self.Y[self.Y.flatten()==1].size
        Y2 = self.Y[self.Y.flatten()==0].size
        Y3 = self.Y[self.Y.flatten()==-1].size
        assert ((Y1 + Y2 == self.Y.size) or (Y1 + Y3 == self.Y.size)), \
            'Inputs should be in {0,1} or {-1,1}.'
        Y_prep[self.Y.flatten() == 0] = -1
        self.Y = Y_prep
        
     def loglikelihood(self, y, f):
        """
        Log Likelihood \\Psi(f) def= \\log p(y | f) + \\log p(f | X)
        
        Rasmussen eq. 3.12

        """
        # log p(y | f):
        lyf = np.log(y*f)
        lpy_f = np.where(y==1, f, 1.-f)
        
        # log p(f | X):
        
        #objective = y*np.log(inv_link_f) + (1.-y)*np.log(inv_link_f)
        p = np.where(y==1, inv_link_f, 1.-inv_link_f)
        return np.log(np.clip(p, 1e-9 ,np.inf))


        