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
        
    def solve(self):
        
        phi = 1/4* (1) 


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
