# Fixed-point iteration for solution of the nonlinear Poisson equation
# u''= alpha f(u), u(-1)=u(1)=0

import numpy.linalg as npla
import numpy as np
import scipy.sparse as sp
from FDLaplacian1D import FDLaplacian1D

alpha = 2.5

# Set up an m by m matrix for FD discretization of the Laplacian
m = 5
K = FDLaplacian1D(-1.0, 1.0, m).todense()

# Set initial guess
u0 = np.ones(m)

# Fixed point iteration
tol = 1.0e-6
maxIter = 100
conv = False

for i in range(maxIter):
  g = alpha*np.cos(u0)
  u1 = npla.solve(K, g)
  print('u[%d]=' % i, u1)
  r = npla.norm(u0 - u1)
  u0 = np.copy(u1)
  if r < m*tol:
    conv = True
    break

if conv:
  print('Converged to solution u=', u0)
else:
  print('Failure to converge after %d iterations' % maxIter)
