import numpy as np
import numpy.linalg as la
import scipy.sparse as sp

def FDCentralDiff1D(a, b, m):
  h = np.abs(b-a)/np.double(m+1)

  aboveDiag = np.ones(m-1)
  belowDiag = -np.ones(m-1)

  D = 0.5/h*sp.diags([belowDiag, aboveDiag], [-1,1])

  return D
