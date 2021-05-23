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
from numpy.random import default_rng

# Write a function to return f(u) and its derivative
def RHSFunc(uk):
  return (np.exp(-uk), -np.exp(-uk))

alpha = 0.5

# Set up an m by m matrix for FD discretization of the Laplacian
m = 500
K = FDLaplacian1D(-1.0, 1.0, m)

# Set initial guess


# RNG
rng = default_rng()

for logsig in range(-16,0):
  sigma = 10**logsig
  tau_r = 1.0e-10
  tau_a = 1.0e-10
  print('residual perturbation: %12.5g' % sigma)
  print('relative tolerance: %12.5g absolute tolerance: %12.5g'
    %(tau_r, tau_a))
  u0 = np.ones(m)


  maxIter = 20 # if it doesn't converge in a few iters, it probably won't ever
  conv = False
  r0 = RHSFunc(u0)[0]
  normR0 = npla.norm(r0)

  # Newton iteration for K u = f(u)
  print('%6s %12s %20s %20s' % ('iter', '|dr|/|r|', '|r|/|r0|', '|du|'))
  for i in range(maxIter):
    fAndDf = RHSFunc(u0)        # Evaluate f(u^k) and f'(u^k)
    f = fAndDf[0]
    df = fAndDf[1]
    J = K - alpha*sp.diags([df],[0])  # Jacobian matrix (discrete Frechet deriv)
    r = K*u0 - alpha*f                # Current residual
    # Perturb the residual
    rPert = np.array([r[j]*(1.0 + rng.normal(scale=sigma)) for j in range(m)])
    pertNorm = npla.norm(rPert - r)/npla.norm(r)

    # Step eqn is: J*v + r = 0, or (K-alpha*diag(f'(u^k)))*v=alpha*f(u^k)-(u^k)''
    newtStep = spla.spsolve(J, -rPert)    # Solve for step
    normR = npla.norm(r)             # Compute residual norm
    normDelta = npla.norm(newtStep)  # Compute step norm
    u0 = u0 + newtStep          # Update solution
    print('%6d %12.5g %20.15g %20.15g' % (i, pertNorm, normR/normR0, normDelta/normR0))
    # Check for convergence: stop if either normR or normDelta is small enough
    if normR < tau_r*normR0 + tau_a:
      conv = True
      break

if conv:
  print('Converged!')
else:
  print('Failure to converge after %d iterations' % maxIter)
