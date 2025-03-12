'''
WHAT IT DOES

2x2 continuous game on [0,1] x [0,1]

- draw contours (curves and filled) of payoff function and, if given, potential function
- draws response graph
- computes interior NE
- plots dynamics and algorithms. Currently implemented:

1. Continuous time DA with entropic regularizer 
2. Continuous time DA with euclidean regularizer (Euclidean projection dynamics)              <---------------   #TO-DO-EUCLIDEAN-CONTINUOUS-ANCHOR    --------------------------------- to check, sharp reduced euclidean metric is NOT identity, cf icml appendix and my notes. Might be missing factor.

3. Discrete time vanilla DA    with euclidean regularizer (Euclidean projection algorithm)    <---------------   #TO-DO-EUCLIDEAN-DISCRETE-ANCHOR    --------------------------------- to check, weird behavior in Prisoner'd Dilemma, 45 degrees in "wrong" direction
4. Discrete time vanilla DA    with entropic  regularizer (Euclidean projection algorithm)
5. Discrete time         DA+   with entropic  regularizer (exponential weights) in extra-gradient variant (not optimistic, two vector queries per step)

# TO DO

-  #TO-DO-EUCLIDEAN-CONTINUOUS-ANCHOR
- #TO-DO-EUCLIDEAN-DISCRETE-ANCHOR
- global careful check and cleanup!
- once all clean, make "the pedagogical drawing", as would use to explain game dynamcis to dad; cf vocal memo
'''



import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import odeint
from scipy import optimize
from aspera import utils
from gamelab import finitegames
import time

# ------------------------------------------------
## PARAMETERS
# ------------------------------------------------

# ------------------------------------------------
## Grid density
# ------------------------------------------------
GRID_DENSITY =  9 # Number of initial conditions for continuous time trajectories, and points where payfield is quivered = square of this number

# If needed, fine-tune for side-to-side potential-harmonic plots
GRID_DENSITY_POTENTIAL =  GRID_DENSITY # Number of trajectories = square of this number; 9 is cool because grid coincides with contours intersections, but it's a bit too dense
GRID_DENSITY_HARMONIC = GRID_DENSITY  # Number of trajectories = square of this number
##################################################

# ------------------------------------------------
## Style
# ------------------------------------------------
PLAYER_1_COLOR = 'black'
PLAYER_2_COLOR = 'black'

REPLICATOR_COLOR = 'red' #'white'
PAYFIELD_COLOR = 'black'

VANILLA_COLOR = 'red' #'white'
EXTRA_COLOR = 'blue'

EUCLIDEAN_COLOR = 'crimson'

POTENTIAL_FUNCTION_COLOR = 'red'

PURE_NE_COLOR = 'cyan'
INITIAL_POINT_COLOR = 'crimson'




# ------------------------------------------------
## Labels
# ------------------------------------------------
CONTINUOUS_TIME_LABEL = 'DA-D'
EXTRAPOLATION_LABEL = 'DA+'
VANILLA_LABEL = 'DA'
PAYOFF_FIELD_LABEL = 'Payoff field'

ENTROPIC_LABEL = ' (entropic)'
EUCLIDEAN_LABEL = ' (euclidean)'

# NE_LABEL = "Strategic center"
PURE_NE_LABEL = "pure NE"
ZERO_PF_LABEL = "zero PF, implies NE"

INCLUDE_LEGEND = 1

INCLUDE_TITLE = True
AXES_LABEL_FONT_SIZE = 8

# ------------------------------------------------
## Contours
# ------------------------------------------------
PLOT_CONTOURS = 1 # Global contours switch


# game type must not be potential to plot contours, can'r remember why... fix this
PLOT_CONTOURS_FIRST_PLAYER = 1
PLOT_CONTOURS_SECOND_PLAYER = 0

PLOT_CONTOURS_POTENTIAL_FUNCTION = 1

PLOT_FILLED_CONTOURS_FIRST_PLAYER = 1
ADD_COLORBAR = 1
DISPLAY_PAYOFF_CONTOURS_VALUES = 0 # tag each contour line with corresponding value

CONTOURS_DENSITY = 10 # number of contours lines
FILLED_CONTOUR_DENSITY = 100 # number of filled contour levels, higher = smoother shades transition

# ------------------------------------------------
## Quivers
# ------------------------------------------------
QUIVER_PAYFIELD = 1
QUIVER_INDIVIDUAL_PAYFIELD = 0
QUIVER_RD = 0

QUIVER_SCALE = 6 # Scaling for quiver plots; high number = short arrow DA is scaled by this number, payfield is scaled by this number SQUARED

# Finetune for potential and harmonic cases if needed
QUIVER_SCALE_POTENTIAL = QUIVER_SCALE # How much smaller potential arrows are then computed number
QUIVER_SCALE_HARMONIC = QUIVER_SCALE # How much smaller harmonic arrows are then computed number

PLOT_SEGMENT_PERPENDICULAR_HARMONIC_CENTER = 0


# ------------------------------------------------
## Continuous time dynamics
# ------------------------------------------------
SOLVE_ODE = 1

# Q_ENTROPY_PARAMETER = 0     # log-barrier
# Q_ENTROPY_PARAMETER = 0.5   # Tsallis entropy
Q_ENTROPY_PARAMETER = 1     # Gibbs entropy (replicator)


ENTROPIC_LABEL += f', q ={Q_ENTROPY_PARAMETER}'

PLOT_CONTINUOUS_DA = 1
INDEX_1 = 1 #4 # choose orbit index, or plot them all
INDEX_2 = 3 # choose orbit index, or plot them all

PLOT_CONTINUOUS_PAYFIELD = 0

RD_LINEWIDTH = 1.5
PF_LINEWIDTH = 0.7

ODE_SOLVER_PRECISION = 1000 # high = more precise

CONTINUOUS_TIME_HORIZON_ENTROPIC = 20 # Max time for dynamical system ODE solver, dual averaging dynamics
CONTINUOUS_TIME_HORIZON_EUCLIDEAN = 0.1 # Max time for dynamical system ODE solver, Euclidean dynamics. Small bc usually hits boundary fast.

# ------------------------------------------------
## Discrete time dynamics
# ------------------------------------------------

# entropic
PLOT_ENTROPIC_VANILLA_FTRL = 0
PLOT_EXTRA_FTRL = 0

STEP_SIZE = 0.3
EXTRA_FTRL_LINEWIDTH = 1.5

# euclidean; not sure it's correct ! careful !
# PLOT_EUCLIDEAN_VANILLA_FTRL = 0 // not implemented for continuous games

TIMESTEPS_EXTRA_FTRL = 25000
TIMESTEPS_VANILLA_FTRL = 1000


# ------------------------------------------------
## Nash equilibria
# ------------------------------------------------
PLOT_NE = 1
EXTEND_DISPLAY_TOLERANCE = 1 # True if want to extend xlim and ylim to see boundary quiver
PLOT_POLAR_CONES = 1






####################################################################################################
############################################# BEGIN CORE ###########################################
####################################################################################################

class Game22():
    def __init__(self, payfuncs, payfield, metric, game_name, game_type = '', potential_function = 0):

        '''
        - all numpy functions take _one_ varible, array of size 2

            def f(vas):
                x, y = vas
                return ...

        - payfuncs is LIST of two functions payfuncs = [u1, u2]
        - each payfunc is numpy function ui( [x1, x2] ) --> R

        - payfield is R2 --> R2 numpy function v( [x1, x2] ) = np.array([v1, v2]) 

        - potential_function is pot( [x1, x2] ) = pot R2 --> R numpy function, if the game is potential it can be used to plot the potential contours

        - metric is string in ['eu', 'sha'], will be used to build continuous dynamics as sharp of reduced payoff field
        - game_name is string used in titles for plots and saved files
        - game type can be set to 'potential' or 'harmonic' to showcase side to side differences, but in general can be left empty
        
        (eg. [ ['Cooperate', 'Defect'], ['cooperate', 'defect'] ]) for prisoner's dilemma that will appear in corners of strategy space
        '''

        # ------------------------------------------------------------------------------------------------------------------------------------------

        # metric 
        assert metric in ['eu', 'sha']
        self.metric = metric
        self.game_type = game_type
        self.is_potential = True if self.game_type == 'potential' else False
        self.is_harmonic = True if self.game_type == 'harmonic' else False
        self.game_name = game_name

        self.num_players = 2
        self.players = [i for i in range(1, self.num_players + 1)]
        
        self.u1, self.u2 = payfuncs
        self.payfield = payfield
        self.potential_function = potential_function

        # --------------------------------------------------------------------
        TR = (1,1) # top right
        TL = (0,1) # top lefy
        BR = (1,0) # bottom right
        BL = (0,0) # bottom left

        self.cones = {      TR : [ np.array([1,0]), np.array([0,1])  ],
                            TL : [ np.array([-1,0]), np.array([0,1]) ],
                            BR : [ np.array([1,0]), np.array([0,-1]) ],
                            BL : [ np.array([-1,0]), np.array([0,-1])]
                        } # corner : benchmark vectors (north, east, south, west)

        # --------------------------------------------------------------------

        self.zero_of_payfield = self.find_zeros_payfield()
        # self.side_NE = self.find_side__NE()
        self.pure_loose_NE, self.pure_strict_NE = self.find_pure_NE()

        # Potential matrix
        if self.is_potential:
            try:
                assert potential_function != 0
            except:
                raise Exception("For potential game need provide potential function")
            self.min_potential_point, self.min_potential_value, self.max_potential_point, self.max_potential_value = utils.optimize_over_square(potential_function)


            


    # --------------------------------------------------------------------
    ## Metrics
    # --------------------------------------------------------------------
    def sha_inv(self, var):
        # Inverse of second derivative of regularizer in entropic (replicator, Shahshahani) case; build dynamics as x' = V#(x)
        # Deprecated in favor of more general h_double_prime approach
        pass
        # return np.array( [xi - xi**2 for xi in var] )

    def h_double_prime(self, var):

        # Second derivative of regularizer; build dynamics as x' = V(x) / h''(x)

        return np.array( [ xi**(Q_ENTROPY_PARAMETER - 2) + (1-xi)**(Q_ENTROPY_PARAMETER - 2) for xi in var] )


    # --------------------------------------------------------------------
    ## Dual averaging continuous time
    # --------------------------------------------------------------------


    def DA(self, var, t):

        # Deprecated in favor of more general h_double_prime approach
        # return  self.sha_inv(var) * self.payfield(var)

        return self.payfield(var) / self.h_double_prime(var)

    # --------------------------------------------------------------------
    ## Payfield Flow (NOT equivalent to euclidean dynamics, payfield leaves feasible space rather than being projected onto.
    # --------------------------------------------------------------------

    def PF(self, var, t):
        return self.payfield(var)

    def PF1_quiver(self, var, t):
        return [ self.payfield(var)[0], 0  ]

    def PF2_quiver(self, var, t):
        return [0, self.payfield(var)[1], 0  ]


    # --------------------------------------------------------------------
    ## DA+ (discrete time), entropic regularizer. Here x is primal variable and y dual variable. Extra-gradient variant (two gradient queries per step)
    # --------------------------------------------------------------------

    def entropic_prox(self, x, y,  step):
        # Entropic regularizer prox map in reduced coordinates, obtained pulling back entropic prox map in usual way.
        num = x * np.exp( step * y )
        den = np.array([1, 1]) + num - x
        return num / den


    def extra_ftrl(self, x0, num_iterations, initial_step):
        # Reduced coordinates. Can be adapted to other regularizer just by changing prox map.
        x = np.array(x0)
        trajectory = [x0]
        step = initial_step

        for t in range(1,num_iterations):
            x_lead = self.entropic_prox(x, self.payfield(x), step)     # extrapolation step
            x = self.entropic_prox(x, self.payfield(x_lead), step)     # main step
            trajectory.append(x)
            # step = step / t                  # update step size

        x1 = [x[0] for x in trajectory]
        x2 = [x[1] for x in trajectory]

        return x1, x2


    def vanilla_entropic_ftrl(self, x0, num_iterations, initial_step):
        # Reduced coordinates. Can be adapted to other regularizer just by changing prox map.
        x = np.array(x0)
        trajectory = [x0]
        step = initial_step

        for t in range(1,num_iterations):
            x = self.entropic_prox(x, self.payfield(x), step)     # main step
            trajectory.append(x)
            # step = step / t                  # update step size

        x1 = [x[0] for x in trajectory]
        x2 = [x[1] for x in trajectory]

        return x1, x2

    # --------------------------------------------------------------------
    ## Vanilla DA (discrete time), Euclidean regularizer. Here x is FULL primal variable and y FULL dual variable. Must work in full coordinates to project on simplex. <--------- #TO-DO-EUCLIDEAN-DISCRETE-ANCHOR --------------------------------------------------
    # --------------------------------------------------------------------

    # def projection_simplex(self, v, z=1):
    #     # Full coordinates
    #     # v is lenght N array to be projected on (N-1) simplex; in this case N = 2
    #     # z I don't know, should be 1
    #     # taken from https://gist.github.com/mblondel/6f3b7aaad90606b98f71
    #     n_features = v.shape[0]
    #     u = np.sort(v)[::-1]
    #     cssv = np.cumsum(u) - z
    #     ind = np.arange(n_features) + 1
    #     cond = u - cssv / ind > 0
    #     rho = ind[cond][-1]
    #     theta = cssv[cond][-1] / float(rho)
    #     w = np.maximum(v - theta, 0)
    #     return w

    # def euclidean_prox(self, x, y, step):
    #     # Euclidean regularizer prox map, full coordinates
    #     # x and y are lenght 4 arrays
    #     # step is scalar

    #     x = np.array(x) # lenght 4 array
    #     y = np.array(y) # lenght 4 array
    #     p = x + step * y       # lenght 4 array, needs be projected on product of two 1-simplices

    #     p1, p2 = p[0:2], p[2:4]

    #     proj_1 = self.projection_simplex( p1 )
    #     proj_2 = self.projection_simplex( p2 )

    #     return np.concatenate( (proj_1, proj_2) ) # length 4 array 


    # def vanilla_euclidean_ftrl(self, x0, num_iterations, initial_step):
    #     # Full coordinates
    #     x = np.array(x0) # size 4
    #     trajectory = [x0] # time series
    #     step = initial_step # scalar

    #     for t in range(1,num_iterations):
    #         x = self.euclidean_prox(x, self.full_payfield(x), step)     # main step
    #         trajectory.append(x)
    #         # step = step / t                  # update step size

    #     # extract first and third out of 
    #     x1 = [x[0] for x in trajectory]
    #     x2 = [x[2] for x in trajectory]

    #     return x1, x2

    # ------------------------------------------------
    ## NE finder
    # ------------------------------------------------



    def find_zeros_payfield(self):
        # zeros of payfield
        root =  optimize.root(self.payfield, [0.5, 0.5]).x
        return root if finitegames.Utils.is_feasible_22(root) else np.nan

    # def find_side__NE(self):

    #     def right_root(y):
    #         return self.payfield( [1, y] )[1]
    #     def left_root(y):
    #         return self.payfield( [0, y] )[1]
    #     def up_root(x):
    #         return self.payfield( [x, 1] )[0]
    #     def down_root(x):
    #         return self.payfield( [x, 0] )[0]

    #     def right_pos(y):
    #         return self.payfield( [1, y] )[0]
    #     def left_neg(y):
    #         return self.payfield( [0, y] )[0]
    #     def up_pos(x):
    #         return self.payfield( [x, 1] )[1]
    #     def down_neg(x):
    #         return self.payfield( [x, 0] )[1]

    #     sign = lambda x: -1 if x < 0 else (1 if x > 0 else 0)

    #     side_roots = [right_root, left_root, up_root, down_root]
    #     side_signs = [right_pos, left_neg, up_pos, down_neg]
    #     side_key =   [+1, -1, +1, -1]

    #     for i, func in enumerate(side_roots):
    #         result = optimize.root_scalar(func, x0 = np.random.rand())

    #     # to finish, cumbersome setup, streamline. Anyways not good because if there is a continuum, solver will find only a pointy. Better approach: since I already quiver payfield on boundary, for every quiver on boundary check if perpendicular to side. cf  anchor-smarter-side-nash
        
            


    def plot_polar_cone(self, ax):
        for corner in self.cones:
            h, v = self.cones[corner]
            p1, p2 = corner + h, corner + v
            points = [corner, p1, p2]

            # fill area of polar cone
            ax.fill( *utils.coords_points( points ), color = 'gray', alpha = 0.1 )

            # plot sides of polar cone
            # ax.quiver(*corner, *h, width=0.002, headlength=3, color = 'gray', alpha = 0.3 )
            # ax.quiver(*corner, *v, width=0.002, headlength=3, color = 'gray',  alpha = 0.3 )



    def find_pure_NE(self):

        loose_NE = [ ]
        strict_NE = [  ]

        def check_NE(point, benches):
            """evaluates V(x) ar corners and checks if belongs to polar cone, by measuring angle beteween benchmark vectors (nourth, east, south, west) vectors
            The method angle_between_vectors always returns a positive angle
            Finds pure NE, strict and non-strict, where payfield does NOT vanish."""
            tolerance = 1e-4

            vector = self.payfield(point)

            # if pure is zero of payfield, it is non-strict NE
            if np.linalg.norm(vector) < tolerance:
                loose_NE.append(point)
                return

            angles = [ utils.angle_between_vectors(vector, bench) for bench in benches ]
            a, b = angles

            if a < tolerance or b < tolerance:
                # if either of the angles is zero, pure nash, non strict (lay on boundary of polar cone)
                # print(f'loose, {point}, {vector}, {angles}')
                loose_NE.append(point)

            elif a < 90 and b < 90:
                # if both angles < 90 deg, strict nash (lay in interior of polar cone)
                # print(f'strict, {point}, {vector}, {angles}')
                strict_NE.append(point)
        # -------------------------------

        for corner in self.cones:
            benches = self.cones[corner]
            check_NE( corner, benches )

        return [loose_NE, strict_NE]

    # --------------------------------------------------------------------
    ## Begin plotting methods
    # --------------------------------------------------------------------

    def contour_plot(self, ax):

        # initialize empty legend elements to return, to avoid error in case this function is called but all the contours flags are False
        legend_elements = []

        # get max and min payoffs to set contours extreme values
        
        # min_point_1, min_value_1, max_point_1, max_value_1 = utils.optimize_over_square(u1)
        # min_point_2, min_value_2, max_point_2, max_value_2 = utils.optimize_over_square(u2)
        
        # utility_levels_1 = np.linspace(min_value_1, max_value_1, CONTOURS_DENSITY + 1)
        # utility_levels_2 = np.linspace(min_value_2, max_value_2, CONTOURS_DENSITY + 1)


        # if self.is_potential:
        #     utility_levels_1, utility_levels_2 = 
        #     contour_plot_levels_start = 1

        # elif self.is_harmonic:
        #     utility_levels_1, utility_levels_2 = 
        #     contour_plot_levels_start = 1

        # else:
        #     utility_levels_1, utility_levels_2 = 
        #     contour_plot_levels_start = 0

            
        y1 = np.linspace(0, 1, 100)
        y2 = np.linspace(0, 1, 100)
        Y1, Y2 = np.meshgrid(y1, y2)


        # label every n countour lines
        add_countour_label_every = 1


        if PLOT_CONTOURS_FIRST_PLAYER or PLOT_FILLED_CONTOURS_FIRST_PLAYER: data_1 = self.u1( (Y1, Y2) )

        # contours player 1
        if PLOT_CONTOURS_FIRST_PLAYER:
            contour_plot_1 = ax.contour(Y1, Y2, data_1, levels = CONTOURS_DENSITY, colors = PLAYER_1_COLOR, linestyles = 'dashed', linewidths = 0.6)
            legend_elements = [matplotlib.lines.Line2D([0], [0], color = PLAYER_1_COLOR, label = 'Payoff contours pl. 1',linestyle = 'dashed', linewidth= 0.6)]
            if DISPLAY_PAYOFF_CONTOURS_VALUES:
                [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in ax.clabel(contour_plot_1, levels = contour_plot_1.levels[contour_plot_levels_start::add_countour_label_every] ,  inline=True, fontsize=7, fmt='(%1.1f)')] # fmt rounds to one decimal place

        # filled contour player 1
        if PLOT_FILLED_CONTOURS_FIRST_PLAYER:

            # sometimes some error with levels = FILLED_CONTOUR_DENSITY * (max_value_1 - min_value_1)
            try:
                payoff_contourf = ax.contourf(Y1, Y2, data_1, cmap = 'viridis', alpha = 0.8, levels = FILLED_CONTOUR_DENSITY )
            except:
                payoff_contourf = ax.contourf(Y1, Y2, data_1, cmap = 'viridis', alpha = 0.8, levels = FILLED_CONTOUR_DENSITY )

            if PLOT_FILLED_CONTOURS_FIRST_PLAYER and ADD_COLORBAR:
                payoff_colorbar = plt.colorbar(payoff_contourf)
                payoff_colorbar.set_label('Payoff player 1 ', labelpad=20, rotation=90,  fontsize = 15) #, va='bottom')

        
        # contours player 2
        if PLOT_CONTOURS_SECOND_PLAYER:

            data_2 = self.u2( (Y1, Y2) ) 
            contour_plot_2 = ax.contour(Y1, Y2, data_2, levels = CONTOURS_DENSITY, colors = PLAYER_2_COLOR, linestyles = 'dashdot', linewidths = 0.6)
            legend_elements.extend( [   matplotlib.lines.Line2D([0], [0], color = PLAYER_2_COLOR, label = 'Payoff contours pl. 2',linestyle = 'dashdot', linewidth = 0.6)] )

            if DISPLAY_PAYOFF_CONTOURS_VALUES:
                [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in ax.clabel(contour_plot_2, levels = contour_plot_2.levels[contour_plot_levels_start::add_countour_label_every] ,  inline=True, fontsize=7, fmt='(%1.1f)')]


        # contours potential function
        if self.is_potential and PLOT_CONTOURS_POTENTIAL_FUNCTION:
            data_potential = self.potential_function( (Y1, Y2) )
            UTILITY_LEVELS_POTENTIAL_FUNCTION = np.linspace(self.min_potential_value, self.max_potential_value, CONTOURS_DENSITY + 1)
            contour_plot_potential = ax.contour(Y1, Y2, data_potential, levels = UTILITY_LEVELS_POTENTIAL_FUNCTION, colors = POTENTIAL_FUNCTION_COLOR, linestyles = 'dashdot', linewidths = 0.6)
            if DISPLAY_PAYOFF_CONTOURS_VALUES:
                [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in ax.clabel(contour_plot_potential, levels = contour_plot_potential.levels[0::add_countour_label_every] ,  inline=True, fontsize=7, fmt='(%1.1f)' )]
            legend_elements.extend( [ matplotlib.lines.Line2D([0], [0], color = POTENTIAL_FUNCTION_COLOR, label = 'Potential function',linestyle = 'dashdot', linewidth = 0.6) ] )

        return legend_elements

        

    def quiver_RD_plot(self, ax):

        # if self.is_potential:
        #     y1, y2 = np.linspace(0, 1, GRID_DENSITY_POTENTIAL), np.linspace(0, 1, GRID_DENSITY_POTENTIAL)

        #     # exclude boundaries
        #     # y1, y2 = y1[1:-1], y2[1:-1]

        # elif self.is_harmonic:
        #     y1, y2 = np.linspace(0, 1, GRID_DENSITY_HARMONIC), np.linspace(0, 1, GRID_DENSITY_HARMONIC)

        # else:
        #     y1, y2 = np.linspace(0, 1, GRID_DENSITY), np.linspace(0, 1, GRID_DENSITY)


        y1, y2 = np.linspace(0, 1, GRID_DENSITY), np.linspace(0, 1, GRID_DENSITY)
        


        Y1, Y2 = np.meshgrid(y1, y2)

        # dual averaging dynamics, continuous time
        DA = self.DA((Y1, Y2), 0)


        # Payfield on grid
        PF = self.PF((Y1, Y2), 0)
        
        if self.is_potential:
            scale = QUIVER_SCALE_POTENTIAL
        elif self.is_harmonic:
            scale = QUIVER_SCALE_HARMONIC
        else:
            scale = QUIVER_SCALE

        # Quiver for each player
        # ax.quiver(Y1, Y2, *RD1, scale=2.5, color = PLAYER_1_COLOR, width=0.005, headlength=5)
        # ax.quiver(Y1, Y2, *RD2, scale=2.5, color = PLAYER_2_COLOR, width=0.005, headlength=5)

        # Quiver dual averaging field
        if QUIVER_RD:
            ax.quiver(Y1, Y2, *DA, scale = scale, color = REPLICATOR_COLOR, width=0.003, headlength=3)
        
        # Quiver payoff field
        if QUIVER_PAYFIELD:
            # can color code here somehow "if payfield on boundary perpendicular to boundary, nash equilibrium" <------------- # anchor-smarter-side-nash
            ax.quiver(Y1, Y2, *PF, scale = scale * scale, color =  PAYFIELD_COLOR, width=0.005, headlength=4)

        if QUIVER_PAYFIELD and QUIVER_INDIVIDUAL_PAYFIELD:

            PF1 = self.PF1_quiver((Y1, Y2), 0)
            PF2 = self.PF2_quiver((Y1, Y2), 0)

            ax.quiver(Y1, Y2, *PF1, scale = scale * scale, color = PLAYER_1_COLOR, width=0.003, headlength=3)
            ax.quiver(Y1, Y2, *PF2, scale = scale * scale, color = PLAYER_2_COLOR, width=0.003, headlength=3)



        # legend_elements = [ ]

        #if self.is_potential:
        # payoff_field_label = 'Payoff field'
        #payoff_field_label = 'Payoff'
        # legend_elements = [ ]
        legend_elements =  [matplotlib.lines.Line2D([0], [0], color= PAYFIELD_COLOR, label = PAYOFF_FIELD_LABEL)] 


        # extra gradient entropic DA
        if PLOT_EXTRA_FTRL:

            # pick an initial point
            initial_point = np.array([ y1[1], y2[3] ])
            print(f'Initial point of extra DA: {initial_point}')
            plt.scatter(*initial_point, color = INITIAL_POINT_COLOR, s = 20)
            EGMD = self.extra_ftrl( initial_point, TIMESTEPS_EXTRA_FTRL, STEP_SIZE )
            nash_x, nash_y = EGMD[0][-1], EGMD[1][-1]

            plt.plot(*EGMD, color = EXTRA_COLOR,  linewidth = EXTRA_FTRL_LINEWIDTH, label = EXTRAPOLATION_LABEL + ENTROPIC_LABEL)
            # plt.scatter( [nash_x], [nash_y], color = 'black', s = 10 )
            legend_elements.extend( [matplotlib.lines.Line2D([0], [0], color = EXTRA_COLOR, label=EXTRAPOLATION_LABEL + ENTROPIC_LABEL)]  )

        # vanilla entropic DA
        if PLOT_ENTROPIC_VANILLA_FTRL:
            initial_point = np.array([ y1[1], y2[3] ])
            print(f'Initial point of entropic vanilla DA: {initial_point}')
            plt.scatter(*initial_point, color = INITIAL_POINT_COLOR, s = 20)
            vanilla_entropic_ftrl_trajectory = self.vanilla_entropic_ftrl( initial_point, TIMESTEPS_VANILLA_FTRL, 0.02 )
            plt.plot(*vanilla_entropic_ftrl_trajectory, color = VANILLA_COLOR, linewidth = 0.7, label = VANILLA_LABEL + ENTROPIC_LABEL)
            legend_elements.extend( [matplotlib.lines.Line2D([0], [0], color = VANILLA_COLOR, label = VANILLA_LABEL  + ENTROPIC_LABEL)]  )


         # vanilla euclidean DA (to check)
        # if PLOT_EUCLIDEAN_VANILLA_FTRL:
            
        #     initial_point = np.array([ y1[1], 1-y1[1], y2[3], 1-y2[3] ])
        #     print(f'Initial point of euclidean vanilla DA: {initial_point}')
        #     plt.scatter(*initial_point, color = INITIAL_POINT_COLOR, s = 20)
        #     print(initial_point)
        #     plt.scatter(initial_point[0], initial_point[2], color = 'black')
        #     EUCLIDEAN_MIRROR_DESCENT = self.vanilla_euclidean_ftrl( initial_point, TIMESTEPS_VANILLA_FTRL, 0.01 )
        #     plt.plot(*EUCLIDEAN_MIRROR_DESCENT, color = EUCLIDEAN_COLOR,  linewidth = 0.7, label = VANILLA_LABEL + EUCLIDEAN_LABEL)
        #     legend_elements.extend( [matplotlib.lines.Line2D([0], [0], color = EUCLIDEAN_COLOR, label = VANILLA_LABEL + EUCLIDEAN_LABEL )]  )

        return legend_elements, y1, y2


    def ode_plot(self, ax):

        legend_elements = []

        # Set up grid

        if self.is_potential:
            y1_range, y2_range = np.linspace(0, 1, GRID_DENSITY_POTENTIAL), np.linspace(0, 1, GRID_DENSITY_POTENTIAL)

            # exclude boundaries
            # y1_range, y2_range = y1_range[1:-1], y2_range[1:-1]
            
        elif self.is_harmonic:
            y1_range, y2_range = np.linspace(0, 1, GRID_DENSITY_HARMONIC), np.linspace(0, 1, GRID_DENSITY_HARMONIC)

        else:
            y1_range, y2_range = np.linspace(0, 1, GRID_DENSITY), np.linspace(0, 1, GRID_DENSITY)

        # Time horizons for ODE trajectories
        t_eu = np.linspace(0, CONTINUOUS_TIME_HORIZON_EUCLIDEAN, ODE_SOLVER_PRECISION)
        t_sha = np.linspace(0, CONTINUOUS_TIME_HORIZON_ENTROPIC, ODE_SOLVER_PRECISION)

        # Starting points for ODE trajectories
        y1 = y1_range
        y2 = y2_range
        start = [ [a,b] for a in y1 for b in y2 ]

        # start = [ [y1[ INDEX_1 ],y2[INDEX_2]]  ]

        # dual averaging
        if PLOT_CONTINUOUS_DA:
            ax.plot(*zip(*odeint(self.DA, start[0], t_sha)), color = REPLICATOR_COLOR, linewidth = RD_LINEWIDTH)
            [ ax.plot(*zip(*odeint(self.DA, p, t_sha)), color = REPLICATOR_COLOR, linewidth = RD_LINEWIDTH) for p in start[1:] ]
            legend_elements.extend( [ matplotlib.lines.Line2D([0], [0], color = REPLICATOR_COLOR, label = CONTINUOUS_TIME_LABEL + ENTROPIC_LABEL ) ])

        # Euclidean
        if PLOT_CONTINUOUS_PAYFIELD:
            ax.plot(*zip(*odeint(self.PF, start[0], t_eu)), color = PAYFIELD_COLOR, linewidth = PF_LINEWIDTH, linestyle = '--')
            [ ax.plot(*zip(*odeint(self.PF, p, t_eu)), color = PAYFIELD_COLOR, linewidth = PF_LINEWIDTH, linestyle = '--') for p in start[1:] ]
            legend_elements.extend( [matplotlib.lines.Line2D([0], [0], color = PAYFIELD_COLOR, label = CONTINUOUS_TIME_LABEL + EUCLIDEAN_LABEL, linestyle = '--', linewidth = PF_LINEWIDTH)] )

        return  legend_elements

    def full_plot(self,ax):

        legend_elements = []



        if PLOT_CONTOURS:
            legend_elements.extend(self.contour_plot(ax))

        if SOLVE_ODE:
            legend_elements.extend(self.ode_plot(ax))

        if QUIVER_PAYFIELD:
            legend_elements_quiver, GRID_1, GRID_2 = self.quiver_RD_plot(ax)
            legend_elements.extend(legend_elements_quiver)

        if PLOT_POLAR_CONES:
            self.plot_polar_cone(ax)


        


        # ax.set_title(f'Shahshahani vs. Euclidean individual gradient ascent \n $2 \\times 2$ {self.game_name} - {self.game_type}', fontsize = '9')
        # ax.set_title(f'Replicartor dynamics and payoff field \n $2 \\times 2$ {self.game_name} - {self.game_type}', fontsize = '10')

        # full_plot_title = f'$2 \\times 2$ {self.game_name} - {self.game_type}'

        externalities_type = 'anti-aligned'
        if self.is_potential:
            externalities_type = 'aligned'


        if INCLUDE_TITLE:
            full_plot_title = self.game_name
            ax.set_title(full_plot_title, fontsize = '12')

        # Boundaries of [0,1] x [0,1]
        # ax.plot([0,0], [0,1], color = 'k', linewidth = 0.5)
        # ax.plot([1,1], [0,1], color = 'k', linewidth = 0.5)
        # ax.plot([0,1], [0,0], color = 'k', linewidth = 0.5)
        # ax.plot([0,1], [1,1], color = 'k', linewidth = 0.5)


        if PLOT_NE:
            for ne in self.pure_strict_NE:
                plt.scatter(ne[0], ne[1], color = PURE_NE_COLOR, zorder=10, s = 100)
                vector = self.payfield(ne)
                plt.quiver( *ne, *vector )

            for ne in self.pure_loose_NE:
                plt.scatter(ne[0], ne[1], edgecolor = PURE_NE_COLOR, facecolor = 'none', zorder=10, s = 400, linewidths = 2)

            if len(self.pure_strict_NE) != 0 or len(self.pure_loose_NE) != 0:
                legend_elements.extend( [matplotlib.lines.Line2D([0], [0], color = PURE_NE_COLOR, label = PURE_NE_LABEL, linestyle = '', marker = 'o' )] )


            try:
                ax.scatter( *self.zero_of_payfield, color = PAYFIELD_COLOR, zorder=10, s = 60)

                # zorder is like z-index in css, higher plots this point on top of other graphical elements. Need high else the contourfill and the extra DA cover it
                legend_elements.extend( [matplotlib.lines.Line2D([0], [0], color = PAYFIELD_COLOR, label = ZERO_PF_LABEL, linestyle = '', marker = 'o' )] )

            except:
                'something wrong trying to scatter zeros of payfield'


        # Hard coded removed axes ticks labels
        # plt.gca().set_xticks([0,0.5,1])
        # plt.gca().set_yticks([0,0.5,1])

        # -------------------------------------------------------------------------
        ## Harmonic: print "perpendicular" vector from center
        # -------------------------------------------------------------------------
        if PLOT_SEGMENT_PERPENDICULAR_HARMONIC_CENTER:
            grid_index_1 = [INDEX_1]
            grid_index_2 = [INDEX_2]

            try:
                for i in range(len(grid_index_1)):
                    plt.plot([self.zero_of_payfield[0], GRID_1[ grid_index_1[i] ]], [self.zero_of_payfield[1], GRID_2[ grid_index_2[i] ]], 'k--' )
            except:
                print("You're trying to plot harmonic center, but there is no interior Nash. Switch PLOT_SEGMENT_PERPENDICULAR_HARMONIC_CENTER to False")

        # -------------------------------------------------------------------------

        if INCLUDE_LEGEND:
            ax.legend(handles=legend_elements, loc='upper left', fontsize = 10)

        
        display_tolerance = 10e-2 if EXTEND_DISPLAY_TOLERANCE else 0
        plt.xlim(0 - display_tolerance, 1 + display_tolerance)
        plt.ylim(0 - display_tolerance, 1 + display_tolerance)
        ax.set_aspect('equal', adjustable='box')

    # ---------------------------------------
    def return_NE_info(self):
        return f"""
        Zero of payfield: {self.zero_of_payfield}\n
        Pure non-strict NE: {self.pure_loose_NE}\n
        Pure strict NE: {self.pure_strict_NE}\n
        """


##################################################################################################

# --------------------------------------------------------
## Bestiario di giochi
# --------------------------------------------------------

# # --------------------------------------------------------
# # Matching pennies
# # --------------------------------------------------------
def u1( vas ):
    x1, x2 = vas
    return 4 * x1 * x2 - 2 * (x1 + x2) + 1

def u2( vas ):
    x1, x2 = vas
    return -(4 * x1 * x2 - 2 * (x1 + x2) + 1)

def v ( vas ):
    x1, x2 = vas
    return np.array( [ 4 * x2 - 2, 2 - 4 * x1 ] )
# # # --------------------------------------------------------

# # # --------------------------------------------------------
# # # strict NE
# # # --------------------------------------------------------
# def u1( vas ):
#     x1, x2 = vas
#     return x1*x2 - 3*x1 + 3

# def u2( vas ):
#     x1, x2 = vas
#     return x1*x2 - x1 + x2 + 1

# def v ( vas ):
#     x1, x2 = vas
#     return np.array( [x2 - 3, x1 + 1])
# # # # --------------------------------------------------------

# # --------------------------------------------------------
# # pure non strict NE
# # --------------------------------------------------------
# def u1( vas ):
#     x1, x2 = vas
#     return x1*x2 - 3*x2 + 3

# def u2( vas ):
#     x1, x2 = vas
#     return x1*x2 + x1 - x2 + 1

# def v ( vas ):
#     x1, x2 = vas
#     return np.array(  [x2, x1 - 1]  )
# # # --------------------------------------------------------


# # --------------------------------------------------------
# # Gen harmonic, continuum of NE on boundary
# # --------------------------------------------------------
# def u1( vas ):
#     x1, x2 = vas
#     return 4*x1*x2 - 2*x1 - 2*x2 + 1

# def u2( vas ):
#     x1, x2 = vas
#     return -2*x1*x2 + 2*x1 - 2*x2 - 1

# def v ( vas ):
#     x1, x2 = vas
#     return np.array( [-2 + 4*x2, -2*x1 + 2]  )
# # # --------------------------------------------------------




# --------------------------------------------------------
# Cournot
""" 
0 < c1, c2 < 1
interior iff, for i different from j, i, j in {1,2}
 2 ci - 1 < cj < 2 ci + 2
"""
# --------------------------------------------------------

c1 = 0.05
# c2 = 0.15
# c2 = 2*c1 + 2
c2 = c1

assert c1 == c2
c = c1


if (2 * c1 - 1 < c2 < 2 * c1 + 2) and (2 * c2 - 1 < c1 < 2 * c2 + 2):
    info_eq = 'Interior equilibrium'
else:
    info_eq = 'No interior equilibrium'

# def u1( vas ):
#     x1, x2 = vas
#     return x1 * (1 - x1 - x2) - c1 * x1

# def u2( vas ):
#     x1, x2 = vas
#     return x2 * (1 - x1 - x2) - c2 * x2


# def v( vas ):
#     x1, x2 = vas
#     return np.array( [ 1 - c1 - 2 * x1 - x2, 1 - c2 - 2 * x2 - x1] )

# def pot( vas ):
#     x1, x2 = vas
#     return x1 * (1 - c1) + x2 * (1 - c2) - x1**2 - x2**2 - x1*x2
# --------------------------------------------------------



u = [u1, u2]
# G = Game22(u, v, 'sha', game_name = f'Cournot - {info_eq}', game_type = 'potential', potential_function = pot)
G = Game22(u, v, 'sha', game_name = f'test', game_type = '')



# --------------------------------------------------------
## Plot
# --------------------------------------------------------

# --------------------------------------------------------
## Default
# --------------------------------------------------------
fig, axs = plt.subplots(1, 1, figsize=(13, 9))#, dpi = 2400)

# --------------------------------------------------------
## Thumbnail
# --------------------------------------------------------

# Desired pixel dimensions
# width_pixels = 320
# height_pixels = 256

# Desired DPI
# dpi = 500  # Adjust as needed for your desired resolution

# Convert pixel dimensions to inches
# width_in_inches = width_pixels / dpi
# height_in_inches = height_pixels / dpi

# Create the figure with the specified size
# plt.figure(figsize=(width_in_inches, height_in_inches))




# fig, axs = plt.subplots(1, 1, figsize=(width_in_inches, height_in_inches))#, dpi = 2400)

# --------------------------------------------------------
G.full_plot(axs)
# axs.scatter( (1-c)/3, (1-c)/3 ) # manual cournot equilibrium
###############################################


# --------------------------------------------------------
## Save methods
# --------------------------------------------------------

root_directory = './Results/'

game_directory = f'{root_directory}/{G.game_name}'
current_directory = f'{game_directory}/{time.time()}'

# utils.make_folder(current_directory)

# plt.savefig(f'{current_directory}/{G.game_name}.pdf', bbox_inches='tight')#, pad_inches = 0)
# utils.write_to_txt(f'{current_directory}/{G.game_name}.txt', u)
# utils.write_to_txt(f'{current_directory}/{G.game_name}.txt', G.return_NE_info())


#####################################################




print(G.return_NE_info())

plt.show()




