# Newton iteration for solution of the nonlinear Poisson equation
# u''=alpha*f(u), u(-1)=u(1)=0.
# Linear equation for the Newton step v will be
# v''= alpha*f'(u^k)*v - r(u^k), with BCs v(-1)=v(1)=0.
# or v'' - alpha*f'(u^k)*v = -r(u^k), where r(u^k)= (u^k)'' - alpha*f(u^k)
# Jacobian is K - alpha*diag(f')
import numpy.linalg as npla
import scipy.sparse.linalg as spla
import numpy as np
import scipy.sparse as sp
from FDLaplacian1D import FDLaplacian1D

# Write a function to return f(u) and its derivative
def RHSFunc(uk):
  return (np.exp(-uk), -np.exp(-uk))

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
  fAndDf = RHSFunc(u0)        # Evaluate f(u^k) and f'(u^k)
  f = fAndDf[0]
  df = fAndDf[1]
  J = K - alpha*sp.diags([df],[0])  # Jacobian matrix (discrete Frechet deriv)
  r = K*u0 - alpha*f                # Current residual
  # Step eqn is: J*v + r = 0, or (K-alpha*diag(f'(u^k)))*v=alpha*f(u^k)-(u^k)'' 
  newtStep = spla.spsolve(J, -r)    # Solve for step
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
