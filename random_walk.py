# -*- coding: utf-8 -*-
"""
Random walker to solve the greens function of poission equation
"""

import numpy as np
import matplotlib.pyplot as plt


class RandomWalk:
    
    def __init__(self,grid_size, spacing, x0, boundary_values, convergance_tolerence, func, ):

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
        self.N = grid_size
        
        self.grid_shape = (self.N,self.N)

        self.h = spacing
        
        self.grid = np.empty( self.grid_shape )
        
        self.grid[:] = boundary_values
        self.grid[1:-1, 1:-1] = x0
        
        np.where(self.grid ==0, 1e-10, self.grid) # prevents potential divide
                                                  # by 0 error
                                                  
    def solve(self,initial_position):
        #Greens grid is used to determine the probabilities 
        #(and thus greens function) of each boundary point bering hit
        greens_grid = self.grid.copy()
                                                  
###############################################################################
# Initial Conditions 
###############################################################################

boundary_values = 5
grid_size = 100
h = 1
x0 = 3

epsilon = 1e-3

test_func = 1

###############################################################################
# Testing
###############################################################################                               
a = RandomWalk(grid_size, h, x0, boundary_values,epsilon,  test_func)