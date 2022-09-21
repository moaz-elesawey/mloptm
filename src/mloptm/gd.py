### Multi Variable Gradient Descent Methods

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


from mloptm.utils import EvalExpr
from mloptm.methods import Golden
from mloptm.exceptions import NotOptimizedError


class SteepestDescent:
    def __init__(self, f, variables):
        self.f = f.replace("^", "**")
        self.variables = variables

        self.expr, self.symbols = EvalExpr(self.f, self.variables)
        self.expr_lambda = sp.lambdify([self.symbols], self.expr)

        self.errors = []
        self.steps = []
        self._minima = None

    @property
    def minima(self):
        return self._minima

    @minima.setter
    def minima(self, val):
        raise ValueError("minima cannot be set. only calculated")

    def Minimize(self, x0, eps=1e-7, verbose=True):
        """
        Actual Minimization Implementation of the Steepest Descent Method.
        
        Parameters
        ----------
         - x0 (list) : the initial point of the minimizing process
         - eps (float) : the tolerance to compare to the error from the minimizing process.

        Rerturns
        --------
         - minima (list) : the minimum value of the function after the minimizing process.
        """

        self.x0 = x0

        ## define the alpha
        alpha = sp.Symbol("alpha", positive=True)
        
        ## compute the gradient and convert to matrix
        grad = sp.Matrix( [self.expr.diff(v) for v in self.symbols] ).n()
        
        ## convert initial point to Matrix
        X0 = sp.Matrix(x0)
        self.steps.append(np.array(X0.T.tolist()[0], dtype=np.float64))
        
        alpha_evaluated = (X0 - alpha * grad).subs([ (var, val) for var, val in zip(self.symbols, x0) ]).n()

        alpha_lambda = sp.lambdify([alpha], self.expr.subs(
                                    [ (var, val) for var, val in 
                                     zip(self.symbols, alpha_evaluated) ]))

        minimum_alpha = Golden(alpha_lambda).Minimize(a0=-1, b0=1, eps=1e-6)

        xk = (X0 - minimum_alpha * grad).subs([ (var, val) for var, val in zip(self.symbols, X0) ]).n()
        self.steps.append(np.array(xk.T.tolist()[0], dtype=np.float64))

        for _ in range(50):
            
            self.steps.append(np.array(xk.T.tolist()[0], dtype=np.float64))
            
            alpha_evaluated = (X0 - alpha * grad).subs([ (var, val) for var, val in zip(self.symbols, X0) ]).n()
            dummy = self.expr.subs([ (var, val) for var, val in zip(self.symbols, alpha_evaluated) ]).n()

            alpha_lambda = sp.lambdify([alpha], dummy, modules="numpy")

            minimum_alpha = Golden(alpha_lambda).Minimize(a0=-1, b0=1, eps=1e-5)

            xk = (X0 - minimum_alpha * grad).subs([ (var, val) for var, val in zip(self.symbols, X0) ]).n()
            
            err = (xk - X0).norm().n() / np.maximum(1, X0.norm().n())

            self.errors.append(err)
            
            if err < eps: break

            if verbose:
                print("Error at Iter [{:>6d}] = {:.12f}".format(_+1, err))
            
            X0 = xk

        self._minima = xk.n().tolist()

        return self._minima

    def PlotError(self, **kw):
        if self.minima is None:
            raise NotOptimizedError("function did not minimized yet.")

        self.errors = np.array(self.errors)
        iterations = range(1, self.errors.shape[0]+1)
        
        fig, ax = plt.subplots(nrows=1, figsize=(10, 6))
        ax.plot(iterations, self.errors, marker="o", color="k",
                markerfacecolor="w", markeredgecolor="k", markersize=8, linewidth=2.)
        ax.set_title("Error Over Iterations", fontsize=15)
        ax.set_xlabel("Iterations", fontsize=15)
        ax.set_ylabel(r"$ \frac{|X^{k+1} - X^k|}{\max(1, |X^k|)} $", fontsize=16)

        plt.show()

        if kw.get("save", None) and kw.get("filename", None):
            fig.savefig(kw.get("filename"), dpi=200)

        return fig

    def PlotContour(self, xdomain, ydomain, **kw):
        if self.minima is None:
            raise NotOptimizedError("function did not minimized yet.")

        if len(self.symbols) != 2:
            raise ValueError("Can only be used to 3D systems, where symbols are just 2.")

        self.steps = np.array(self.steps, dtype=np.float64)
        self.x_steps = self.steps[:, 0]
        self.y_steps = self.steps[:, 1]

        xmid, ymid = self.x0

        # xs = np.linspace( xmid-xmid*2, xmid+xmid*2, 100 )
        # ys = np.linspace( ymid-ymid*2, ymid+ymid*2, 100 )

        xs = np.linspace( *xdomain, 100 )
        ys = np.linspace( *ydomain, 100 )

        xx, yy = np.meshgrid(xs, ys)
        zz = self.expr_lambda([xx, yy])

        fig, ax = plt.subplots(nrows=1, figsize=(9, 7))
        cs = ax.contourf(xx, yy, zz, cmap="viridis", alpha=0.7, levels=14)
        ax.plot(self.x_steps, self.y_steps, marker="o", color="k",
                markerfacecolor="w", markeredgecolor="k", markersize=7, linewidth=1.5)
        ax.plot(self.x_steps[-1], self.y_steps[-1], marker="o", markerfacecolor="r",
                markeredgecolor="k", markersize=15, alpha=0.4)
        ax.set_title("Steepest Descent Minimization")
        ax.set_xlabel(self.symbols[0])
        ax.set_ylabel(self.symbols[1])
        cbar = plt.colorbar(cs)
        cbar.ax.set_ylabel("Z Values")
        # ax.axis("square")

        plt.show()

        if kw.get("save", None) and kw.get("filename", None):
            fig.savefig(kw.get("filename"), dpi=200)

        return fig

    def Plot3D(self, xdomain, ydomain, **kw):
        if self.minima is None:
            raise NotOptimizedError("function did not minimized yet.")

        if len(self.symbols) != 2:
            raise ValueError("Can only be used to 3D systems, where symbols are just 2.")

        self.steps = np.array(self.steps, dtype=np.float64)
        self.x_steps = self.steps[:, 0]
        self.y_steps = self.steps[:, 1]

        xmid, ymid = self.x0

        xs = np.linspace( *xdomain, 100 )
        ys = np.linspace( *ydomain, 100 )

        xx, yy = np.meshgrid(xs, ys)
        zz = self.expr_lambda([xx, yy])

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection="3d")

        ax.plot_surface(xx, yy, zz, cstride=6, rstride=6, cmap="coolwarm", alpha=0.5)
        ax.contour(xx, yy, zz, zdir='z', offset=np.min(zz), cmap="coolwarm")

        ax.plot(self.x_steps, self.y_steps, self.expr_lambda([self.x_steps, self.y_steps]), marker="o",
                color="k", markerfacecolor="w", markeredgecolor="k", markersize=7, linewidth=1.5)

        ax.plot3D(self.x_steps[-1], self.y_steps[-1], self.expr_lambda([self.x_steps[-1], self.y_steps[-1]]), 
                marker="o", markerfacecolor="r", markeredgecolor="k", markersize=15, alpha=0.4)

        ax.set_title("Steepest Descent Minimization")
        ax.set_xlabel(self.symbols[0])
        ax.set_ylabel(self.symbols[1])
        ax.set_zlabel(f"f({self.symbols[0]},{self.symbols[1]})")

        plt.show()

        if kw.get("save", None) and kw.get("filename", None):
            fig.savefig(kw.get("filename"), dpi=200)

        return fig


class Newton:
    def __init__(self, f, variables):
        pass

    def Minimize(self, x0, eps=1e-7):
        pass

    def ErrorPlot(self):
        pass

    def ContourPlot(self, domain, **kw):
        pass

