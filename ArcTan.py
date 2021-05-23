import numpy as np

# Function for which to find root
def func(x):
  return np.arctan(x)

# First derivative
def dFunc(x):
  return 1.0/(1.0 + x*x)


# Unmodified Newton's method
def newton1D(f, x0, maxIters=20, tol=1.0e-14):

  # Compute initial residual
  r = f(x0)
  # Store it for use in convergence tests
  rInit = np.abs(r)

  # Main loop
  for i in range(maxIters):
    print('newton iter=%6d, x=%12.5g, rel resid=%12.5g' % (i,x0,np.abs(r/rInit)))
    # Compute derivative (residual has already been computed)
    df = dFunc(x0)
    # Compute full Newton step
    dx = -r/df
    # Take full Newton step
    x0 = x0 + dx
    # Compute residual
    r = f(x0)
    # Check for convergence
    if np.abs(r) <= tol*rInit:
      print('Newton\'s method converged in %d iters' % i)
      print('solution is x=', x0)
      print('residual is r=', np.abs(r))
      return (x0, True)

  # If exited loop, the method has failed to converge
  print('Failure to converge!')
  return (x0, False)

# Newton's method with simple backtracking
def newtonBacktrack1D(f, x0, maxIters=20, maxBack=20, tol=1.0e-14):

  # Compute initial residual
  r0 = f(x0)
  rInit = np.abs(r0)

  # Main loop
  for i in range(maxIters):
    print('newton iter=%6d, x=%12.5g, rel resid=%12.5g' % (i,x0,np.abs(r0/rInit)))
    # Compute derivative
    df = dFunc(x0)
    # Compute full Newton step (using stored residual)
    dx = -r0/df

    # Set up for line search to reduce residual
    alpha = 1.0
    lsGood = False
    print('Line search:')
    # Line search loop
    for j in range(maxBack):
      # Take trial step
      x1 = x0 + alpha*dx
      # Compure new residual
      r1 = f(x1)
      # Find relative residual compared to that at start of step
      red = np.abs(r1/r0)
      print('\tback=%6d, alpha=%12.5g, reduction=%12.5g' % (j,alpha,red))
      # Have we reduced the residual?
      if red < 1.0: # Yes, accept trial step
        print('Line search found sufficient reduction')
        x0 = x1
        r0 = r1
        lsGood = True # Mark as accepted
        break
      else: # No, reduce step further
        alpha /= 2.0

    if not lsGood: # Detect line search failure
      print('Line search failed!')
      return (x0, False)

    # Check for convergence to root
    if np.abs(r0) <= tol*rInit:
      print('Newton\'s method converged in %d iters' % i)
      print('solution is x=', x0)
      print('residual is r=', np.abs(r0))
      return (x0, True)

  print('Failure to converge!')
  return (x0, False)

if __name__=='__main__':

  print('='*60)
  print('Unmodified Newton')
  (x, conv) = newton1D(func, 10.0)

  print('='*80)
  print('Newton with backtracking line search')
  (x, conv) = newtonBacktrack1D(func, 10.0)
