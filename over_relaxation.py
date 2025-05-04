# Licenced with Monzilla Public Licence 2.0
"""
Successive Over-Relaxiation Method
"""

import numpy as np
import matplotlib.pyplot as plt


class OverRelaxation:
    
    def __init__(self,grid_size, spacing, x0, boundary_values):

        """
        
        Parameters
        ----------
        grid_shape : The size of N to create a NxN grid
                     
        spacing : the grid spacing h 
        
        x0 : Initial conditions of the solution eg. initial charge distribution
             for poission equation.
        
        boundary_values : the boundary values 

        Returns
        -------
        None.

        """
        self.N = grid_size
        
        self.grid_shape = (self.N,self.N)

        self.h = spacing
        
        self.grid = np.empty( self.grid_shape )
        
        self.grid[1:-1,:] = boundary_values
        self.grid[:,1:-1] = boundary_values
        
        self.grid[1:-1, 1:-1] = x0
        
        #fixing grid corners having numbers like 1e-300
        self.grid[0,0] =0
        self.grid[-1,-1] =0
        self.grid[-1,0] =0
        self.grid[0,-1] =0


    def solve(self, func):
        
        phi = self.grid.copy()
        
        convergance_tolerence = 1
        omega = 2/(1+np.sin(np.pi/self.N))

        while convergance_tolerence >= 0.1:
            old_phi = phi.copy()
            print('sadkjonf')

            for xi in range(1, len(self.grid[:,-1])-1):
                
                for yj in range(1, len(self.grid[:-1, -1]-1)):
                    print(xi)
                    print(yj)
                
                    phi[xi,yj] = omega*( func(xi,yj) + 1/4* (phi[xi,yj-1]+ phi[xi,yj-1]+ phi[xi-1,yj] + phi[xi+1,yj]) ) + (1-omega)*old_phi[xi,yj]
            


            convergance_tolerence -=0.1
    
        return phi

    def plot3d(self):
        ax = plt.axes(projection = "3d")
        ax.scatter(3,5,7)
        plt.show()
        
        return

###############################################################################
# Initial Conditions 
###############################################################################

boundary_values = 3
grid_size = 100
h = 1
x0 = 50

###############################################################################
# Testing
###############################################################################


a = OverRelaxation(grid_size, h, x0, boundary_values)

def test_func(x,y):
    return np.sin(x)+np.sin(y)

b = a.solve(test_func)
print(b[1,1])

c = a.plot3d()

