import numpy as np


x0 = -1.0
tol = 1.0e-16
maxIter = 100
conv = False

print('%3d %20.16g %12.4g' % (0, x0, x0))
for i in range(maxIter):
  x1 = np.exp(-x0)
  r = np.abs(x0 - x1)
  x0 = x1
  print('%3d %20.16g %12.4g' % (i, x1, r))
  if r < tol:
    conv = True
    break

if conv:
  print('Converged to solution x=%g' % x0)
else:
  print('Failure to converge after %d iterations' % maxIter)
