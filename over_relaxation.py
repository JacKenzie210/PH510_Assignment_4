# Licenced with Monzilla Public Licence 2.0
"""
Successive Over-Relaxiation Method
"""

import numpy as np
import matplotlib.pyplot as plt


class OverRelaxation:
    
    def __init__(self,grid_shape, spacing, x0):
        

        self.h = spacing
        
        self.grid = np.empty( grid_shape )
        
        self.grid[1:-1,:] = boundary_values
        self.grid[:,1:-1] = boundary_values
        
        self.grid[1:-1, 1:-1] = x0
        
        # test = self.grid[0, 1:-1]

        # print(test)
        # print(np.shape(test))
    def solve(self):
        
        phi = self.grid
        convergance_tolerence = 1
        while convergance_tolerence >= 0.1:
            print('sadkjonf')
            
            for xi in range(len(self.grid[0, 1:-1])):
                for yj in range(len(self.grid[0, 1:-1])):
                    
                
                    phi[xi,yj] = 1/4* (phi[xi,yj-1]+ phi[xi,yj-1]+ phi[xi-1,yj] + phi[xi+1,yj]) 
            
            

            convergance_tolerence -=0.1

    def plot3d(self):
        return

###############################################################################
# Initial Conditions 
###############################################################################

boundary_values = 1
shape = (16,16)
h = 1
x0 = 0

###############################################################################
# Testing
###############################################################################

a = OverRelaxation(shape, h, x0)


b = a.solve()