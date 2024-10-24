import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as sp
x, y = sp.symbols('x, y', positive=True)


class LinFTRL():
    """docstring for Akin"""
    def __init__(self, simkernels, kerlabels, radius):
        super(LinFTRL, self).__init__()

        self.radius = radius

        # kernel functions and their derivatives as symbolic expressions
        self.simkernels = simkernels
        self.simkernels1 = [sp.diff(theta, x) for theta in self.simkernels]
        self.simkernels2 = [sp.diff(theta1, x) for theta1 in self.simkernels1]

        # kernel functions and their derivatives as regular functions
        self.kernels = [sp.lambdify(x, theta) for theta in self.simkernels]
        self.kernels1 = [sp.lambdify(x, theta1) for theta1 in self.simkernels1]
        self.kernels2 = [sp.lambdify(x, theta2) for theta2 in self.simkernels2]

        # names of kernel functions
        self.kerlabels = kerlabels

        # Regularizers R --> R
        self.simregularizers = [theta + theta.subs(x, self.radius - x) for theta in self.simkernels]
        self.simregularizers1 = [sp.diff(h, x) for h in self.simregularizers]
        self.simregularizers2 = [sp.diff(h1, x) for h1 in self.simregularizers1]

        self.regularizers =  [sp.lambdify(x, h) for  h in self.simregularizers]
        self.regularizers1 = [sp.lambdify(x, h1) for h1 in self.simregularizers1]
        self.regularizers2 = [sp.lambdify(x, h2) for h2 in self.simregularizers2]

        # Choice map = inverse of h' = (h*)'
        self.simchoices = [ sp.solve( h1 - y, x )[0] for h1 in self.simregularizers1 ]
        self.choices = [sp.lambdify(y, Q) for Q in self.simchoices]



        # Hessian metric: g = hess(h) = h''
        self.simmetrics = [sp.simplify(g) for g in self.simregularizers2]
        self.metrics = [sp.lambdify(x, g) for g in self.simmetrics]

        # Sharps = g inverse
        self.simsharps = [sp.simplify(1 / g) for g in self.simmetrics] 
        self.sharps = [sp.lambdify(x, sh) for sh in self.simsharps]

        epsilon = 1e-5

        # positve reals
        self.R = np.linspace(epsilon , self.radius - epsilon, 200 ) # positive reals
        
        # positve orthant of R2
        # self.C = np.meshgrid(self.R, self.R) # positive orthant of R2

        # positve reals, higher resolution (for plotting)
        self.R_hd = np.linspace(epsilon , self.radius - epsilon, 800 )

        # positve orthant of R2, higher resolution (for plotting)
        # self.C_hd = np.meshgrid(self.R_hd, self.R_hd)

        print("hello world!")

    def round_ex(self, expr):
        """Round floats in sympy expressions"""
        
        for a in sp.preorder_traversal(expr):
            if isinstance(a, sp.Float):
                expr = expr.subs(a, round(a, 2))
        return expr

    def plot_method(self, functions, simfunctions, **kwargs):
        """ Plots a tuple of functions : R --> R, used to plot in the same figure various kernels, their derivatives, Akin isometries, etc."""

        xDomain = kwargs['xDomain']

        plt.figure(figsize=kwargs.get('figsize', (8, 6)))  # Pass figsize as kwarg; if none, default size is (8, 6) inches

        
        for i, f in enumerate(functions):

            # handle euclidean case: in this case f(x) = const = 1 acting on array returns scalar, se plot method gives mismatch in dimension
            if f(0.1) == 1 and f(0.2) == 1 and f(0.3) == 1:
                plt.plot(xDomain, np.ones(len(xDomain)), label = f"{self.kerlabels[i]}: ${sp.latex(  self.round_ex(simfunctions[i])  )}$ " )

            else:
                plt.plot(xDomain, f(xDomain), label = f"{self.kerlabels[i]}: ${sp.latex(  self.round_ex(simfunctions[i])  )}$ " )

            
        plt.legend(loc = 'best', fontsize=12)
        
        plt.title(kwargs['title'])
        plt.xlabel( kwargs['xlabel'] )
        plt.ylabel( kwargs['ylabel'] )
                   
        # plot axes
        plt.plot(plt.xlim(), [0,0], '--k', lw = 0.5)
        plt.plot([0,0], plt.ylim(), '--k', lw = 0.5)



