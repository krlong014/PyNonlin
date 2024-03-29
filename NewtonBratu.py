# Newton iteration for solution of the Bratu equation
# u''=exp(-u), u(-1)=u(1)=0

import numpy.linalg as npla
import scipy.sparse.linalg as spla
import numpy as np
import scipy.sparse as sp
from FDLaplacian1D import FDLaplacian1D

alpha = 0.5

# Set up an m by m matrix for FD discretization of the Laplacian
m = 5
K = FDLaplacian1D(-1.0, 1.0, m)

# Set initial guess
u0 = -np.ones(m)

# Newton iteration
tol = 1.0e-15
maxIter = 20 # if it doesn't converge in a few iters, it probably won't ever
conv = False

# Newton iteration for K u = f(u)
for i in range(maxIter):
  f = alpha*np.exp(-u0)       # f(u^k)
  J = K + sp.diags([f],[0])   # Jacobian matrix (discrete Frechet deriv)
  r = f - K*u0                # Current residual
  newtStep = spla.spsolve(J, r)    # Solve for step
  normR = npla.norm(r)             # Compute residual norm
  normDelta = npla.norm(newtStep)  # Compute step norm
  u0 = u0 + newtStep          # Update solution
  print('u[%d]=' % i, u0)
  # Check for convergence: stop if either normR or normDelta is small enough
  if normR < m*tol or normDelta < m*tol:
    conv = True
    break

if conv:
  print('Converged to solution u=', u0)
else:
  print('Failure to converge after %d iterations' % maxIter)
