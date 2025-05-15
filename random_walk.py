# -*- coding: utf-8 -*-
"""
Random walker to solve the greens function of poission equation
"""

import numpy as np
import matplotlib.pyplot as plt
from monte_carlo import MonteCarlo

class GreensFunc:
    """
    GreensFunc initialises a grid (phi)
    has functions std_green - calculated standard deviation of the Laplace
                              Greens grid
                 random_walker - calculates the Greens grid probabilities for
                                 Laplace and Poission.
    """

    def __init__(self, grid_size, spacing, x_0, boundary_values, func):

        """

        Parameters
        ----------
        grid_size : The length of the NxN grid

        spacing : the grid spacing h

        self.x_0 : Initial conditions of the solution eg. initial charge distribution
             for poission equation.

        boundary_values : the boundary values. can be single value or size 4
                          array corrosponding to upper, lower, left and right.

        func: the function f(x,y).

        Returns
        -------
        None.

        """
        self.h_space = spacing
        self.n_grid_points = int(grid_size // self.h_space) # determines total size with respect
                                          # to spacing.

        self.grid_shape = (self.n_grid_points, self.n_grid_points)

        self.grid = np.ones( self.grid_shape )
        self.func = func
        self.bound = boundary_values

        self.x_0 = x_0
        if np.size(self.x_0) > 1 : #if a constant gradient array i.e task 4.bb
            self.x_0 = self.grid[1:-1,1:-1]
            for i in range(len(self.x_0[0])):
                self.x_0[:,i] = x_0
        else:
            self.x_0 = x_0
        if  np.size(boundary_values) == 4:

            self.grid[0] = boundary_values[0]
            self.grid[:,0] = boundary_values[1]
            self.grid[-1] = boundary_values[2]
            self.grid[:,-1] = boundary_values[3]

        else:
            self.grid*boundary_values


        self.grid[1:-1, 1:-1] = self.x_0

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
        #(and thus greens function) of each boundary point being hit from the
        #initial point
        self.greens_grid = np.zeros( self.grid_shape )

        if initial_position is None:
            initial_position = (len(self.greens_grid)//2 , len(self.greens_grid)//2)

        initial_position = initial_position // self.h_space
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


        if self.x_0 == 0 : # less computationally demanding to record only the
                     # boundary values of the walk when considering Laplace.

            for _ in range(n_walks):

                x_i = int(initial_position[0])
                y_j = int(initial_position[1])

                while not ( x_i ==0 or x_i ==len(self.greens_grid)-1
                           or y_j == 0 or y_j ==len(self.greens_grid)-1):

                    direction = np.random.randint(0,100)

                    if 0 <= direction <= up-1 :
                        x_i -= 1
                        #print(f'({x_i},{y_j}) U')
                    elif up <= direction <= up+down-1:
                        x_i +=1
                        #print(f'({x_i},{y_j}) D')

                    elif up + down <= direction <= up+down+left-1:
                        y_j -= 1
                        #print(f'({x_i},{y_j}) L')
                    else:
                        y_j +=1
                        #print(f'({x_i},{y_j}) R')


                self.greens_grid[x_i,y_j]+=1
                #print(f'({x_i},{y_j})')

            sum_of_hits = np.sum(self.greens_grid)

            if sum_of_hits == n_walks:
                print(f'All {n_walks} walks accounted for and normalised')

            #Turning self.greens_grid from hits into probabilities
            self.greens_grid = self.greens_grid/n_walks

        else: # counts every grid space for Poission equation where
              # charges != 0
            n_steps = 0
            for i in range(n_walks):
                x_i = int(initial_position[0])
                y_j = int(initial_position[1])

                while not ( x_i ==0 or x_i ==len(self.greens_grid)-1
                           or y_j == 0 or y_j ==len(self.greens_grid)-1):

                    direction = np.random.randint(0,100)

                    if 0 <= direction <= up-1 :
                        x_i -= 1
                        self.greens_grid[x_i,y_j]+=1
                        n_steps +=1
                        #print(f'({x_i},{y_j}) U')
                    elif up <= direction <= up+down-1:
                        x_i +=1
                        self.greens_grid[x_i,y_j]+=1
                        n_steps +=1
                        #print(f'({x_i},{y_j}) D')

                    elif up + down <= direction <= up+down+left-1:
                        y_j -= 1
                        self.greens_grid[x_i,y_j]+=1
                        n_steps +=1
                        #print(f'({x_i},{y_j}) L')
                    else:
                        y_j +=1
                        self.greens_grid[x_i,y_j]+=1
                        n_steps +=1
                        #print(f'({x_i},{y_j}) R')

                self.greens_grid[x_i,y_j]+=1
                n_steps +=1

                #print(f'({x_i},{y_j})')

            sum_of_steps = int(np.sum(self.greens_grid))

            if sum_of_steps == n_steps:
                print(f'All {n_steps} steps accounted for and normalised')
            else:
                print('not normalised')

            #Turning self.greens_grid from total steps into probabilities
            self.greens_grid = self.greens_grid/n_steps

        self.std_greens_val = self.std_greens(self.greens_grid)
        # self.greens_grid[0] /= self.h_space**2
        # self.greens_grid[:,0] /= self.h_space**2
        # self.greens_grid[-1]  /= self.h_space**2
        # self.greens_grid[:,-1] /= self.h_space**2
        return self.greens_grid, self.std_greens_val



    def solve(self, initial_position = None, n_walks = None,
              direction_probs = None):

        if not hasattr(self, 'greens_grid'):#prevent recalculation
                                                 #if already conducted
            g_grid = self.random_walker(initial_position, n_walks,
                                       direction_probs)
            self.greens_grid  =g_grid[0]
            self.std_greens_val = g_grid[1]

        if self.x_0 == 0:

            self.phi_grid = self.grid.copy()
            self.phi_ij = np.sum (self.greens_grid*self.phi_grid)
            std_phi = self.std_greens_val

        else:
            x_0_copy = self.x_0
            self.x_0 = 0 #aquires the Laplace Greens function portion
            self.random_walker(initial_position,n_walks,direction_probs)

            self.phi_grid = self.grid.copy()
            self.phi_ij = np.sum (self.greens_grid*self.phi_grid)
            std_green = self.std_greens_val


            self.x_0 = x_0_copy #returns to origional value for G Poission.
            self.random_walker(initial_position,n_walks,direction_probs)
            #grid_coords = self.grid_coords(self.greens_grid)
            integrand_green = np.mean(self.greens_grid)

            #implementing Monte Carlo
            rand_x = np.random.uniform(0, self.bound , size =self.n_grid_points)
            mc_arr = np.array([rand_x])
            monte_carlo = MonteCarlo(mc_arr,[0,self.n_grid_points*self.h_space])
            monte_carlo_integral = monte_carlo.integrate(self.func)
            self.phi_ij += integrand_green*monte_carlo_integral

            mc_stats = monte_carlo.mean_var_std(self.func)

            std_phi = np.sqrt(std_green**2 + mc_stats[1] )

        return self.phi_grid, self.phi_ij, std_phi





    def grid_coords(self, grid):

        """
        generation of a meshgrid to create arrays representing the co-ords
        of the inputted grid. Allows for easy index_ing for f(x,y) in the class.
        """

        #This generates the co-ordinates for the grid positions
        row, column = np.indices((self.n_grid_points, self.n_grid_points))
        coords = np.vstack([row.ravel(), column.ravel()])

        return coords*self.h_space


    # def plot3d(self,initial_position = None,
    #            n_walks = None,
    #            direction_probs = None):


    #     if not hasattr(self, 'self.phi_ij'): #runs the solve function to plot data
    #         self.solve(initial_position,     # if not already done so
    #                    n_walks,
    #                    direction_probs)

    #     x = np.linspace(0, self.n_grid_points , self.n_grid_points)
    #     y = np.linspace(0, self.n_grid_points , self.n_grid_points)
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
if __name__ == "__main__":
#     boundary_values = 1
#     grid_size = 10
#     h = 0.1
#     x0 = 1

#     epsilon = 1e-3

    def test_func(coords):
        return np.sin(np.sum(coords))**2

#     ipos = (grid_size // 2, grid_size//(2))
#     n_walks = 100

#     prob_walks = np.array([0.25,0.25,0.25,0.25])
###############################################################################
# Testing
###############################################################################
# if __name__ == "__main__":

#     test = GreensFunc(grid_size, h, x0, boundary_values, test_func)

#     test_green = test.random_walker(ipos, n_walks, prob_walks )

#     test_solve = test.solve(ipos, n_walks, prob_walks)

#     test_grid_mc =test.grid_coords(test_green)









###############################################################################
# Task 3
###############################################################################
if __name__ == "__main__":
    grid_size_t3 = 10e-2 #m
    h_t3 = 1e-3 # m  or 1e-1 cm
    x0_t3 = 1 #using poissons Green functions.
    boundary_t3 = 3
    n_walks_t3 = 100
    point_a = (5e-2,5e-2) #cm
    point_b = (2.5e-2,2.5e-2) #cm
    point_c = (0.1e-2, 2.5e-2) #cm
    point_d = (0.1e-2, 0.1e-2) #cm

    points_t3 = np.array([point_a,point_b,point_c,point_d])
    task_3_sol = np.zeros(len(points_t3))
    t3_std = np.zeros(len(points_t3))


    for i in range(len(points_t3)):
        initialise = GreensFunc(grid_size_t3,h_t3,x0_t3,boundary_t3, test_func)
        t3_walker = initialise.random_walker(points_t3[i],n_walks_t3)
        t3_grid = t3_walker[0]
        t3_std[i]= t3_walker[1]

        fig, ax = plt.subplots()
        imshow = ax.imshow(t3_grid, cmap="coolwarm")
        ax.set_title(f"greens function at point {points_t3[i]/grid_size_t3} cm")
        ax.set_xlabel('y grid_point (cm/spacing)')
        ax.set_ylabel('x grid_point (cm/spacing)')
        fig.colorbar(imshow)

    print('\nTask 3\n-------\nsee graphs')
    for j in range(len(points_t3)):
        print(f'std = {t3_std[j]}')



###############################################################################
# Task 4 Laplace
###############################################################################

if __name__ == "__main__":

    print('\nTask 4\n-------\n')
    #taking the perameters from task 3
    grid_size_t4 = 10e-2 #m
    h_t4 = 1e-3 # m  or 1e-1 cm
    x0_t4 = 0
    n_walks_t4 = 100

    prob_walks = np.array([0.25,0.25,0.25,0.25])
    points_t4 = points_t3.copy()
    def function(coords):
        return 0
    print('Laplace grid')
    def task_4_bounds(x0_task4):
        # part a
        boundary_t4 = 1

        task_4a = GreensFunc(grid_size_t4, h_t4, x0_t4, boundary_t4, function)
        task_4a_greens = task_4a.random_walker(points_t4[0], n_walks_t4)
        task_4a_solve = task_4a.solve(points_t4[0], n_walks_t4)
        print(f'a) Phi_ij = {task_4a_solve[1]} +/- {task_4a_solve[2]}')

        #part b
        boundary_t4 = np.array([1,1,-1,-1])

        task_4b = GreensFunc(grid_size_t4, h_t4, x0_t4, boundary_t4, function)
        task_4b_greens = task_4b.random_walker(points_t4[0], n_walks_t4)
        task_4b_solve = task_4b.solve(points_t4[0], n_walks_t4)
        print(f'b) Phi_ij = {task_4b_solve[1]} +/- {task_4b_solve[2]}')

        #part c
        boundary_t4 = np.array([2,2,1,-4])

        task_4c = GreensFunc(grid_size_t4, h_t4, x0_t4, boundary_t4, function)
        task_4c_greens = task_4c.random_walker(points_t4[0], n_walks_t4)
        task_4c_solve = task_4c.solve(points_t4[0], n_walks_t4)
        print(f'c) Phi_ij = {task_4c_solve[1]} +/- {task_4c_solve[2]}')
        #part d
        boundary_t4 = np.array([2,2,1,-4])

        task_4c = GreensFunc(grid_size_t4, h_t4, x0_t4, boundary_t4, function)
        task_4c_greens = task_4c.random_walker(points_t4[0], n_walks_t4)
        task_4c_solve = task_4c.solve(points_t4[0], n_walks_t4)
        print(f'b) Phi_ij = {task_4c_solve[1]} +/- {task_4c_solve[2]}')

        return

    def task_4_charge(t4_bounds):
        print('Poission grid')
        #part d
        x0_d = 10 # C
        task_4d = t4_bounds(x0_d)

        evals = np.linspace(1,0,grid_size_t4) #C
        x0_e = n
        task_4e = t4_bounds(x0_e)




    # def task_4(points_array):



    #     solution = 1
    #     return solution