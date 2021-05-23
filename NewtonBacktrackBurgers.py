# Newton iteration for solution of the Burgers equation
# -u_xx + beta u u_x - f = 0 , u(-1)=u(1)=0,
# with simple backtracking for globalization.
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


beta = 20.0


# Set up an m by m matrix for FD discretization of the Laplacian
m = 50
K = FDLaplacian1D(-1.0, 1.0, m)
D = FDCentralDiff1D(-1.0, 1.0, m)
f = np.ones(m)
# Set initial guess
u0 = 0.0*np.ones(m)
r = -K*u0 + beta*np.multiply(u0, D*u0) - f
normR0 = npla.norm(r)
normR = normR0


# Newton iteration
tol = 1.0e-12
maxIter = 40 # if it doesn't converge in a few iters, it probably won't ever
maxBack = 20
conv = False

# Newton iteration for K u = f(u)
for i in range(maxIter):
  print('newton iter=%6d, rel resid=%12.5g' %(i, normR/normR0))
  # Form Jacobian
  J = -K + beta*sp.diags([D*u0],[0]) + beta*sp.diags([u0],[0])*D
  # Form residual
  r = -K*u0 + beta*np.multiply(u0, D*u0) - f
  # Step eqn is: J*v + r = 0
  newtStep = spla.spsolve(J, -r)    # Solve for step
  normR = npla.norm(r)             # Compute residual norm

  # Do half-step backtracking until a residual decrease is detected
  alpha = 1.0
  lineSearchGood = False
  print('Line search')
  print('\t%10s %12s %s' % ('ls step', 'alpha', 'resid reduction'))
  for j in range(maxBack):
    u1 = u0 + alpha*newtStep
    r1 = -K*u1 + beta*np.multiply(u1, D*u1) - f
    normR1 = 0.9999*npla.norm(r1)
    print('\t%10d %12.8g %g' % (j,alpha,normR1/normR))
    if normR1 < normR: # Decrease detected, end backtracks
      lineSearchGood = True
      u0 = u1.copy()
      break
    alpha /= 3.0

  if not lineSearchGood:
    print('line search failed')

  if normR1 < tol*normR0:
    conv = True
    break

if conv:
  print('Converged to solution u=', u0)
  print('Residual: absolute |r|=%12.5g, relative |r|=%12.5g'%(normR1, (normR1/normR0) ))
else:
  print('Failure to converge after %d iterations' % maxIter)
