import numpy as np
from Tab import Tab

class FDDifferentiator:
    def __init__(self, order=1, hFactor=1):

        self.eps = 1.022e-16
        self.p = order
        self.h = hFactor*self.eps**(1/(self.p+1))
        self.hFactor = hFactor

        stencils = {
        1 : [(0,1),(-1,1)],
        2 : [(-1,1),(-0.5,0.5)],
        4 : [(-2,-1,1,2), (1/12, -2/3, 2/3, -1/12)],
        6 : [(-3, -2, -1, 1, 2, 3),
             (-1/60, 3/20, -3/4, 3/4, -3/20, 1/60)],
        8 : [(-4,-3,-2,-1,1,2,3,4),
            (1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280)],
        3 : [(0,1,2,3), (-11/6, 3, -3/2, 1/3)],
        5 : [(0,1,2,3, 4, 5), (-137/60, 5, -5, 10/3, -5/4, 1/5)]
        }

        stencil = stencils[self.p]
        self.dx = [dx*self.h for dx in stencil[0]]
        self.w = [w/self.h for w in stencil[1]]

    def __str__(self):
        return 'FDDiff(p=%d, h=%12.5g)' % (self.p, self.h)

    def deriv(self, f, x):
        df = 0.0
        for dx_i, w_i in zip(self.dx, self.w):
            f_i = f(x + dx_i)
            df += w_i * f_i
        return df

    def deriv2(self, f, x):
        return (f(x+self.h) + f(x-self.h) - 2.0*f(x))/self.h**2







class FDNewtonSolver1D:
    def __init__(self, maxIters=20, tau_a=1.0e-14, tau_r=1.0e-14,
                diff=FDDifferentiator(1)):

        self.maxIters = maxIters
        self.tau_a = tau_a
        self.tau_r = tau_r
        self.diff = diff

    def solve(self, f, xInit):
        tab0 = Tab()
        tab1 = Tab()

        print(tab0, ''*60)
        print(tab0, 'Starting Newton solver:')
        print(tab1, 'Max iters = ', self.maxIters)
        print(tab1, 'tau_r=%12.5g, tau_a=%12.5g' % (self.tau_r, self.tau_a))
        print(tab1, 'diff=', self.diff)

        # Initialize first step
        x0 = xInit
        f0 = f(x0)
        r0 = np.abs(f0)
        dx = 1.0

        # Run loop
        print('\n', tab0, 'Newton loop')

        for i in range(self.maxIters):
            if i>0:
                f0 = f(x0)
            r = np.abs(f0)
            print(tab1, 'iter %6d x=%20.15g r=%20.15g r/r0=%20.15g dx=%12.5g' %
                (i+1, x0, r, r/r0, dx))

            if r <= self.tau_r * r0 + self.tau_a:
                print(tab0, 'Converged!')
                print(tab1, 'iter %6d x=%25.15g r=%25.15g r/r0=%25.15g' %
                    (i+1, x0, r, r/r0))
                return x0

            df = self.diff.deriv(f, x0)
            ddf = self.diff.deriv2(f, x0)
            c1 = 0.5*np.abs(ddf)/np.abs(df)
            p = self.diff.p
            c2 = (self.diff.h)**(p/(1+p)) / np.abs(df)
            print(tab1, '            c1=%12.5g, c2=%12.5g\n' % (c1, c2))
            dx = -f0/df
            x0 = x0 + dx

        print('Newton solver failed to converge!')
        return x0


if __name__=='__main__':

    def f(x):
        return x*x*x*x - 1/4

    def df(x):
        return 4.0*x*x*x

    x0 = 1.0
    exact = (1/4)**(1/4)

    print('='*80)
    print('Testing FD varying p')
    for p in [1, 2, 3, 4, 5, 6, 8]:
        diff = FDDifferentiator(order=p)
        h = diff.h
        expected = h**p
        dfFD = diff.deriv(f, exact)
        dfEx = df(exact)
        err = np.abs(dfFD - dfEx)
        print('p=%4d h=%10.5g h^p=%10.5g df/dx(FD)=%20.15g, df/dx(exact)=%20.15g err=%12.5g' %
            (p, h, expected, dfFD, dfEx, err))

    print('\n\n','='*80)
    print('Testing FD varying h factor')
    p=1
    for hfact in [10**q for q in range(-5,6)]:
        diff = FDDifferentiator(order=p, hFactor=hfact)
        h = diff.h
        expected = h**p
        dfFD = diff.deriv(f, exact)
        dfEx = df(exact)
        err = np.abs(dfFD - dfEx)
        print('p=%4d h=%10.5g h^p=%10.5g df/dx(FD)=%20.15g, df/dx(exact)=%20.15g err=%12.5g' %
            (p, h, expected, dfFD, dfEx, err))


    print('\n\n','='*80)
    print('Testing Newton\s method with FD, varying p')
    for p in [1, 8]:
        diff = FDDifferentiator(order=p)
        newt = FDNewtonSolver1D(diff=diff)

        root = newt.solve(f, x0)
        print('Found root: ', root)
        print('Error is ', np.abs(root-exact))


    print('\n\n','='*80)
    print('Testing Newton\s method with FD, varying hFactor')
    for hfact in [10**q for q in [-6,0,6]]:
        diff = FDDifferentiator(order=1, hFactor=hfact)
        newt = FDNewtonSolver1D(diff=diff)

        root = newt.solve(f, x0)
        print('Found root: ', root)
        print('Error is ', np.abs(root-exact))
