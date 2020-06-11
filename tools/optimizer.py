"""
    file: optimizer.py

    implements optimizers for use in optimization
"""
import numpy as np

class AdamUpdater:
    """AdamUpdater
    the step method updates the variable using the adam optimizer
    
    w = w - lr * (m_hat)/v

    default values taken from PyTorch implementation
    """
    def __init__(self,betas=(0.9,0.99)):
        self._betas = betas
        self._m = None
        self._v = None 
        self._t = 0
        self._epsilon = 1e-8

    @property
    def m(self):
        return self._m

    def step(self,x,lr,df):
        """performs adam update on variable x
        
        Inputs:
            x  -- np column array to be updated
            lr -- learning rate scalar
            df -- gradient vector np column array
        
        Outputs:
            x -- new value of x
        """
        # initialization if first step
        if self._t == 0:
            self._m = np.zeros_like(x)
            self._v = np.zeros_like(x)

        # increment the step
        self._t += 1

        # update m and v and their averages 
        self._m = self._betas[0] * self._m + (1 - self._betas[0]) * df
        self._v = self._betas[1] * self._v + (1 - self._betas[1]) * np.power(df,2)
        m_hat = self._m / (1 - np.power(self._betas[0],self._t))
        v_hat = self._v / (1 - np.power(self._betas[1],self._t))

        # update x
        x -= lr * m_hat /(np.sqrt(v_hat) + self._epsilon)


