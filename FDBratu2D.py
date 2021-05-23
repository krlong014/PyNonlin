import scipy.sparse as sp
import numpy as np
from FDLaplacian2D import FDLaplacian2D

class FDBratu2D:
    def __init__(self, m=4, alpha=0.5):
        self.m = m
        self.alpha = alpha

        self.A = -FDLaplacian2D(-1.0, 1.0, m)

    def initialU(self):
        return np.ones(self.m*self.m)

    def evalF(self, u):
        return self.A*u - self.alpha*np.exp(-u)

    def evalJ(self, u):
        J = self.A.copy()
        g = self.alpha*np.exp(-u)
        d = J.diagonal()
        J.setdiag(d+g)

        return J
