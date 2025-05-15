# Licenced with Monzilla Public Licence 2.0
"""
Successive Over-Relaxiation Method
"""

import numpy as np
import matplotlib.pyplot as plt


class OverRelaxation:
    
    def __init__(self,grid_size, spacing, x0, boundary_values, convergance_tolerence, func):

        """

        Parameters
        ----------
        grid_size : The size of N to create a NxN grid
                     
        spacing : the grid spacing h 
        
        x0 : Initial conditions of the solution eg. initial charge distribution
             for poission equation.
        
        boundary_values : the boundary values 
        
        func: the function f(x,y). 

        Returns
        -------

        None.

        """
        
        self.h = spacing
        if grid_size % 2 ==0: # this will create an odd sized grid such that a 
            grid_size += 1    # centre poiont will be present 
            
        self.N = int(grid_size // h)
        self.grid_shape = (self.N, self.N)

        self.grid = np.empty( self.grid_shape )
        
        self.grid[:] = boundary_values
        self.grid[1:-1, 1:-1] = x0
        
        np.where(self.grid ==0, 1e-10*convergance_tolerence, self.grid) # prevents potential divide
                                                  # by 0 error
        
        self.func = func

        self.conv_tolerence = convergance_tolerence

    def solve(self):
        
        self.phi = self.grid.copy()
        
        omega = 2/(1+np.sin(np.pi/self.N))
        
        #initialises the convergance change and ensures its > tolerance
        convergance_change = self.conv_tolerence + 1

        while convergance_change > self.conv_tolerence:
            old_phi = self.phi.copy()

            for xi in range(1, self.N-1):

                for yj in range(1, len(self.grid[:-1, -1]-1)):

                    self.phi[xi,yj] = omega*((h**2 * self.func(xi,yj)) + 1/4* (
                        self.phi[xi,yj-1]+ self.phi[xi,yj+1]+ self.phi[xi-1,yj]
                        + self.phi[xi+1,yj]) ) + ((1-omega)*old_phi[xi,yj])


            convergance_change = np.max( np.abs( (old_phi - self.phi)/old_phi ) )

            print(f'''-----------------------------------------------------------
                  \nOld {np.max(old_phi)}, New {np.max(self.phi)}''',
                  f'\nConvergance = {convergance_change}')

        return self.phi

    def plot3d(self):

        if not hasattr(self, 'self.phi'): #runs the solve function to plot data
            self.solve()                  # if not already done so

        x = np.linspace(0, self.N , self.N)
        y = np.linspace(0, self.N , self.N)
        X, Y = np.meshgrid(x, y)

        ax = plt.axes(projection = "3d")
        ax.plot_surface(X, Y, self.phi, cmap="coolwarm")
        plt.title(r"surface of $\phi$")
        plt.xlabel("x value")
        plt.ylabel("y value")
        ax.set_zlabel(r"$\phi$(x,y)")
        plt.show()

        return

###############################################################################
# Initial Conditions 
###############################################################################


grid_size = 10
boundary_values = 1
h = 1/2
x0 = 3

epsilon = 1e-2

###############################################################################
# Testing
###############################################################################
def test_func(x,y):
    return np.sin(x+y)**2


a = OverRelaxation(grid_size, h, x0, boundary_values,epsilon,  test_func)



b = a.solve()


a.plot3d()

