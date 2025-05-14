# -*- coding: utf-8 -*-
"""
Random walker to solve the greens function of poission equation
"""

import numpy as np
import matplotlib.pyplot as plt
from monte_carlo import MonteCarlo


class RandomWalk:
    
    def __init__(self, grid_size, spacing, x0, boundary_values, func):

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
        self.N = int(grid_size // self.h) # determines total size with respect
                                          # to spacing.
        self.grid_shape = (self.N, self.N)

        self.grid = np.ones( self.grid_shape ) 
        self.func = func
        self.bound = boundary_values
        
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
        self.greens_grid = np.zeros( self.grid_shape )
        
        if initial_position is None:
            initial_position = (len(self.greens_grid)//2 , len(self.greens_grid)//2)
        
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
    
                while not ( xi ==0 or xi ==len(self.greens_grid)-1 or yj == 0 or yj ==len(self.greens_grid)-1):
    
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


                self.greens_grid[xi,yj]+=1
                print(f'({xi},{yj})')

            sum_of_hits = np.sum(self.greens_grid)
              
            if sum_of_hits == n_walks:
               print(f'All {n_walks} walks accounted for and normalised')
                
            #Turning self.greens_grid from hits into probabilities
            self.greens_grid = self.greens_grid/n_walks

        else: # counts every grid space for Poission equation where 
              # charges != 0
              n_steps = 0
              for i in range(n_walks):
      
                  xi = initial_position[0]
                  yj = initial_position[1]

                  while not ( xi ==0 or xi ==len(self.greens_grid)-1 or yj == 0 or yj ==len(self.greens_grid)-1):

                      direction = np.random.randint(0,3)
      
                      if 0 <= direction <= up-1 :
                          xi -= 1
                          self.greens_grid[xi,yj]+=1
                          n_steps += 1

                      elif up <= direction <= up+down-1:
                          xi +=1
                          self.greens_grid[xi,yj]+=1
                          n_steps +=1
                          
                      elif up + down <= direction <= up+down+left-1:
                          yj -= 1
                          self.greens_grid[xi,yj]+=1
                          n_steps +=1

                      elif up+down+left <= direction <= up+down+left+right-1:
                          yj +=1
                          self.greens_grid[xi,yj]+=1  
                          n_steps +=1

                  self.greens_grid[xi,yj]+=1
                  n_steps +=1

                  print(f'({xi},{yj})')

              sum_of_steps = np.sum(self.greens_grid)
 
              if sum_of_steps == n_steps:
                 print(f'All {n_steps} steps accounted for and normalised')

              #Turning self.greens_grid from total steps into probabilities
              self.greens_grid = self.greens_grid/n_steps

        std_greens_grid = self.std_greens(self.greens_grid)

        return self.greens_grid, std_greens_grid


    # def solve(self,initial_position = None, n_walks = None, direction_probs = None):


    #     #solves the Laplace (which Poisson depends on if solving that)
    #     g_grid = self.random_walker(initial_position,n_walks,
    #                                 direction_probs)
    #     phi_grid = self.grid.copy()
    #     self.phi_ij_Laplace = np.sum (g_grid[0]*phi_grid)
    #     std_phi = g_grid[1]

    #     if x0 !=0: #Will describe the solution to the Poission equation (i.e x0!= 0)

    #         g_grid = self.random_walker(initial_position,n_walks,
    #                                     direction_probs)

    #         phi_grid = self.grid.copy()
    #         self.phi_ij = np.sum(g_grid[0]*phi_grid) + (self.h**2/n_walks * 
    #                                                     np.sum(g_grid[0]) )
    #         std_phi = g_grid[1]        

    #         return


    #     return self.phi_ij, std_phi




    def solve(self, initial_position = None, n_walks = None,
              direction_probs = None):

        if not hasattr(self, 'greens_grid'):#prevent recalculation 
                                                 #if already conducted
            self.greens_grid  = self.random_walker(initial_position, n_walks, 
                                       direction_probs)[0]



        grid_coords = self.grid_coords(self.greens_grid)
        
        #implementing Monte Carlo
        #monte = MonteCarlo(grid_coords, [1,self.bound]).integrate(self.func)
        


        integrand_green = np.mean(grid_coords)#self.greens_grid[grid_coords[0],grid_coords[1]]

        solution = integrand_green*( 
            MonteCarlo(grid_coords,[0,self.N*self.h]).integrate(self.func) )
        return solution





    def grid_coords(self, grid):
        
        """
        generation of a meshgrid to create arrays representing the co-ords
        of the inputted grid. Allows for easy indexing for f(x,y) in the class.
        """

        #This generates the co-ordinates for the grid positions       
        row, column = np.indices((self.N, self.N))
        coords = np.vstack([row.ravel(), column.ravel()])

        return coords*self.h
        

    # def plot3d(self,initial_position = None,
    #            n_walks = None,
    #            direction_probs = None):


    #     if not hasattr(self, 'self.phi_ij'): #runs the solve function to plot data
    #         self.solve(initial_position,     # if not already done so
    #                    n_walks, 
    #                    direction_probs)     

    #     x = np.linspace(0, self.N , self.N)
    #     y = np.linspace(0, self.N , self.N)
    #     X, Y = np.meshgrid(x, y)

    #     ax = plt.axes(projection = "3d")
    #     ax.plot_surface(X, Y, self.phi_ij, cmap="coolwarm")
    #     plt.title(r"surface of $\phi$")
    #     plt.xlabel("x value")
    #     plt.ylabel("y value")
    #     ax.set_zlabel(r"$\phi$(x,y)")
    #     plt.show()

    #     return


###############################################################################
# Initial Conditions 
###############################################################################

boundary_values = 5
grid_size = 10
h = 1
x0 = 0

epsilon = 1e-3

def test_func(coords):
    return 1


ipos = (grid_size // (h*2), grid_size//(h*2))
n_walks = 100

prob_walks = np.array([0.25,0.25,0.25,0.25])
###############################################################################
# Testing
###############################################################################  
if __name__ == "__main__":
                                 
    test = RandomWalk(grid_size, h, x0, boundary_values, test_func)
    
    test_green = test.random_walker(ipos, n_walks, prob_walks )
    
    test_solve = test.solve(ipos, n_walks, prob_walks)
    
    test_grid_mc =test.grid_coords(test_green)









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