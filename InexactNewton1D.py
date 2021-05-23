import numpy as np
from numpy.random import default_rng



def func(x):
  df = 4.0*x**3
  f = x**4-1.0

  return (f,df)

rng = default_rng()

for logsig in range(-16, -1):

  sigma = 10**logsig

  tau_r = 1.0e-10
  tau_a = 1.0e-10
  print('residual perturbation: %12.5g' % sigma)
  print('relative tolerance: %12.5g absolute tolerance: %12.5g'
    %(tau_r, tau_a))
  maxIters = 20
  x0 = 1.5
  r0,df = func(x0)

  for i in range(maxIters):
    f,df = func(x0)
    solveErr = rng.normal(scale=sigma)
    step = -f/df*(1.0 + solveErr)
    print('%6d %20.15g %20.15g' % (i, x0, f))
    x0 = x0 + step
    if np.abs(f) <= tau_r*r0 + tau_a:
      print('Converged!\nRoot is %12.15g' % x0)
      break
