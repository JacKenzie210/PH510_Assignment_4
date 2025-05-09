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
        self.h = spacing
        self.N = int(grid_size // h)
        self.grid_shape = (self.N, self.N)

        self.grid = np.empty( self.grid_shape )
        
        self.grid[:] = boundary_values
        self.grid[1:-1, 1:-1] = x0
        
        np.where(self.grid ==0, 1e-10*convergance_tolerence, self.grid) # prevents potential divide
                                                  # by 0 error
        
        self.func = func

        self.conv_tolerence = convergance_tolerence
                                                  
    def random_walker(self,initial_position = None, n_walks = None, direction_probs = None):
        """
        The random walker used to determine the Greens function
        
        Parameters
        ----------        
        initial_position : the co-ordinate of where to begin random  walks
        
        n_walks = number of walks. Used to determine number of times the while 
                  loop is ran.
        
        direction_probs : 1D list (of length = 4) of Probabilities from going 
                         up/down/left/right respectively
        """
        #Greens grid is used to determine the probabilities 
        #(and thus greens function) of each boundary point bering hit
        greens_grid = np.zeros( self.grid_shape )
        
        if initial_position is None:
            initial_position = (len(greens_grid)//2 , len(greens_grid)//2)
        
        if n_walks is None:
            #Defaulting number of walks to 100 if no imput is presented
            n_walks = 100

        
        if direction_probs is None:
            #assigns equal probability of each direction occuring
            direction_probs = np.array([0.25,
                                       0.25,
                                       0.25,
                                       0.25])
        elif np.sum(direction_probs) != 1:
            raise ValueError("ensure sum of probabilities = 1")
        
        up = direction_probs[0] * 100
        down = direction_probs[1] * 100
        left = direction_probs[2] * 100
        right = direction_probs[3] * 100
        

        
        for i in range(n_walks):
            
            xi = initial_position[0]
            yj = initial_position[1]      
            
            while not ( xi ==0 or xi ==len(greens_grid)-1 or yj == 0 or yj ==len(greens_grid)-1):
                
                direction = np.random.randint(0,100)
                
                if 0 <= direction <= up-1 :
                    xi -= 1 
                    print(f'({xi},{yj}) U')
                elif up <= direction <= up+down-1:
                    xi +=1
                    print(f'({xi},{yj}) D')
                    
                elif up + down <= direction <= up+down+left-1:
                    yj -= 1
                    print(f'({xi},{yj}) L')
                elif up+down+left <= direction <= up+down+left+right-1:
                    yj +=1
                    print(f'({xi},{yj}) R')
  
            greens_grid[xi,yj]+=1
            print(f'({xi},{yj})')
        
        sum_of_hits = np.sum(greens_grid)

        if sum_of_hits == n_walks:
            print(f'All {n_walks} walks accounted for and normalised')
         
        #Turning greens_grid from hits into probabilities
        greens_grid = greens_grid/n_walks
        
        return greens_grid


    def solve(self,initial_position = None, n_walks = None, direction_probs = None):

        g_grid = self.random_walker(initial_position,n_walks,direction_probs)
        phi_grid = self.grid.copy()
        phi_ij = np.sum (g_grid*phi_grid)
        
        return phi_ij


###############################################################################
# Initial Conditions 
###############################################################################

boundary_values = 5
grid_size = 10
h = 1
x0 = 3

epsilon = 1e-3

test_func = 1

ipos = (grid_size // 2, grid_size//2)
n_walks = 100

prob_walks = np.array([0.25,0.25,0.25,0.25])
###############################################################################
# Testing
###############################################################################  
if __name__ == "__main__":
                                 
    test = RandomWalk(grid_size, h, x0, boundary_values,epsilon,  test_func)
    
    test_green = test.random_walker(ipos, n_walks, prob_walks )
    
    test_solve = test.solve(ipos, n_walks, prob_walks)