# Newton iteration for solution of the Burgers equation
# -u_xx + beta u u_x - f = 0 , u(-1)=u(1)=0.
#
# The linear equation for the Newton step v will be
#
# -v_xx + beta u^(k) v_x + beta v u^(k)_x - u^(k)_xx + beta u^(k) u^(k)_x-f=0
# J v + r = 0
# J = (-K + beta*diag(u^(k))*D + beta*diag(u^(k)))
# r = -K*u^(k) + beta*diag(u^(k) .* (D*u^(k))) - f

import numpy.linalg as npla
import scipy.sparse.linalg as spla
import numpy as np
import scipy.sparse as sp
from FDLaplacian1D import FDLaplacian1D
from FDCentralDiff1D import FDCentralDiff1D


beta = 100.0


# Set up an m by m matrix for FD discretization of the Laplacian
m = 10
K = FDLaplacian1D(-1.0, 1.0, m)
D = FDCentralDiff1D(-1.0, 1.0, m)
f = np.ones(m)
# Set initial guess
u0 = 0.0*np.ones(m)
r = -K*u0 + beta*np.multiply(u0, D*u0) - f
normR = npla.norm(r)
normR0 = normR

# Newton iteration
tol = 1.0e-14
maxIter = 20 # if it doesn't converge in a few iters, it probably won't ever
conv = False

# Newton iteration for K u = f(u)
for i in range(maxIter):
  print('newton iter=%6d rel resid=%12.5g' % (i, normR/normR0))
  # Form Jacobian
  J = -K + beta*sp.diags([D*u0],[0]) + beta*sp.diags([u0],[0])*D
  # We've computed residual already.

  # Step eqn is: J*v + r = 0
  newtStep = spla.spsolve(J, -r)    # Solve for step

  # Update solution
  u0 = u0 + newtStep
  # Compute residual at new iterate
  r = -K*u0 + beta*np.multiply(u0, D*u0) - f
  normR = npla.norm(r)             # Compute residual norm
  # Check for convergence: stop if either normR or normDelta is small enough
  if normR < tol*normR0:
    conv = True
    break

if conv:
  print('Converged to solution u=', u0)
  print('Residual: absolute |r|=%12.5g, relative |r|=%12.5g'
    %(normR, (normR/normR0) ))
else:
  print('Failure to converge after %d iterations' % maxIter)
