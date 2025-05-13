# -*- coding: utf-8 -*-
"""
Random walker to solve the greens function of poission equation
"""

import numpy as np
import matplotlib.pyplot as plt


class RandomWalk:
    
    def __init__(self, grid_size, spacing, x0, boundary_values):

        """
        
        Parameters
        ----------
        grid_size : The length of the NxN grid
                     
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
        self.N = int(grid_size // self.h) # determines total size with respect to 
                                     # spacing
        self.grid_shape = (self.N, self.N)

        self.grid = np.ones( self.grid_shape ) 
        
        if  np.size(boundary_values) >1:
            print('boundary == ',len(boundary_values))
            print('grid ==', len(self.grid[0]))

            self.grid[0] = boundary_values
            self.grid[:,0] = boundary_values
            self.grid[-1] = boundary_values
            self.grid[:,-1] = boundary_values
            
        else:
            self.grid*boundary_values
        

        self.grid[1:-1, 1:-1] = x0
        
        np.where(self.grid ==0, 1e-12, self.grid) # prevents potential divide
                                                  # by 0 error

    def std_greens(self, grid):
        up_bound = grid[0, 1:-1]
        low_bound = grid[-1, 1:-1]
        left_bound = grid[1:-1, 0]
        right_bound =  grid[1:-1,-1]
        
        total_bounds = np.concatenate([up_bound,low_bound,left_bound,right_bound])         
        
        std = np.std(total_bounds)
                
        return std
                                                  
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
            raise ValueError("ensure sum of probabilities = 1 (decimal probs)")
        
        #converts from decimal to percentage
        up = direction_probs[0] * 100
        down = direction_probs[1] * 100
        left = direction_probs[2] * 100
        right = direction_probs[3] * 100


        if x0 == 0 : # less computationally demanding to record only the 
                     # boundary values of the walk when considering Laplace.
                
            for i in range(n_walks):
    
                xi = initial_position[0]
                yj = initial_position[1]      
    
                while not ( xi ==0 or xi ==len(greens_grid)-1 or yj == 0 or yj ==len(greens_grid)-1):
    
                    direction = np.random.randint(0,100)
    
                    if 0 <= direction <= up-1 :
                        xi -= 1
                        #print(f'({xi},{yj}) U')
                    elif up <= direction <= up+down-1:
                        xi +=1
                        #print(f'({xi},{yj}) D')
                        
                    elif up + down <= direction <= up+down+left-1:
                        yj -= 1
                        #print(f'({xi},{yj}) L')
                    elif up+down+left <= direction <= up+down+left+right-1:
                        yj +=1
                        #print(f'({xi},{yj}) R')                
      
                greens_grid[xi,yj]+=1
                print(f'({xi},{yj})')
                
            sum_of_hits = np.sum(greens_grid)
              
            if sum_of_hits == n_walks:
               print(f'All {n_walks} walks accounted for and normalised')
                
                
        else: # counts every grid space for Poission equation where 
              # charges != 0
              n_steps = 0
              for i in range(n_walks):
      
                  xi = initial_position[0]
                  yj = initial_position[1]
                  
                  
      
                  while not ( xi ==0 or xi ==len(greens_grid)-1 or yj == 0 or yj ==len(greens_grid)-1):
      
                      direction = np.random.randint(0,100)
      
                      if 0 <= direction <= up-1 :
                          xi -= 1
                          greens_grid[xi,yj]+=1
                          n_steps += 1
                      elif up <= direction <= up+down-1:
                          xi +=1
                          greens_grid[xi,yj]+=1
                          n_steps +=1
                          
                      elif up + down <= direction <= up+down+left-1:
                          yj -= 1
                          greens_grid[xi,yj]+=1
                          n_steps +=1

                      elif up+down+left <= direction <= up+down+left+right-1:
                          yj +=1
                          greens_grid[xi,yj]+=1  
                          n_steps +=1

                  greens_grid[xi,yj]+=1
                  n_steps +=1
                  
                  print(f'({xi},{yj})')
              
        
              sum_of_steps = np.sum(greens_grid)
                
              if sum_of_steps == n_steps:
                 print(f'All {n_walks} walks accounted for and normalised')
         
        #Turning greens_grid from hits into probabilities
        greens_grid = greens_grid/n_walks
          
        std_greens_grid = self.std_greens(greens_grid)

        return greens_grid, std_greens_grid


    def solve(self,initial_position = None, n_walks = None, direction_probs = None):
        
        
        if np.sum(self.grid[1:-1, 1:-1]) == 0: # Automatically selects the Laplace 
                                       # solution if all grid points = 0 
            g_grid = self.random_walker(initial_position,n_walks,direction_probs)
            phi_grid = self.grid.copy()
            phi_ij = np.sum (g_grid[0]*phi_grid)
            std_phi = g_grid[1]
            
        else: #Will describe the solution to the Poission equation (i.e x0!= 0)
            return

        
        return phi_ij, std_phi


###############################################################################
# Initial Conditions 
###############################################################################

boundary_values = 5
grid_size = 10
h = 1
x0 = 1

epsilon = 1e-3

test_func = 1

ipos = (grid_size // 2, grid_size//2)
n_walks = 100

prob_walks = np.array([0.25,0.25,0.25,0.25])
###############################################################################
# Testing
###############################################################################  
if __name__ == "__main__":
                                 
    test = RandomWalk(grid_size, h, x0, boundary_values)
    
    test_green = test.random_walker(ipos, n_walks, prob_walks )
    
    test_solve = test.solve(ipos, n_walks, prob_walks)









###############################################################################
# Task 3
# ###############################################################################
# grid_size_t3 = 10e-2 #m
# h_t3 = 1e-4 # m or 1e-2 cm 
# x0_t3 = 0 #using Lapace as this task only considers Green functions.
# boundary_t3 = np.linspace(0, grid_size_t3, int(grid_size_t3//h_t3))
# n_walks_t3 = 10
# point_a = (5e-2,5e-2) #m
# point_b = (2.5e-2,2.5e-2) #m
# point_c = (0.1e-2, 2.5e-2) #m
# point_d = (0.1e-2, 0.1e-2) #m

# points = np.array([point_a,point_b,point_c,point_d])
# task_3 = np.zeros(len(points))
# for i in range(len(points)):
    
#     task_3 = RandomWalk(grid_size_t3,h_t3,x0_t3,boundary_t3)
#     greens_t3 = task_3.random_walker(points[i], n_walks_t3)
#     greens_t3_sol = greens_t3[0]
#     std_t3 = greens_t3[1]
#     print(f'Task 3\n-------\ngreens_function of point {i} = {greens_t3}')


