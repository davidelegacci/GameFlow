'''
WHAT IT DOES

2x2 game finite normal form game

- draw contours (curves and filled) of payoff function and, if given, potential function
- draws response graph
- computes interior NE
- plots dynamics and algorithms. Currently implemented:

1. Continuous time FTRL with entropic regularizer (replicator dynamics)
2. Continuous time FTRL with euclidean regularizer (Euclidean projection dynamics)              <---------------   #TO-DO-EUCLIDEAN-CONTINUOUS-ANCHOR    --------------------------------- to check, sharp reduced euclidean metric is NOT identity, cf icml appendix and my notes. Might be missing factor.

3. Discrete time vanilla FTRL    with euclidean regularizer (Euclidean projection algorithm)    <---------------   #TO-DO-EUCLIDEAN-DISCRETE-ANCHOR    --------------------------------- to check, weird behavior in Prisoner'd Dilemma, 45 degrees in "wrong" direction
4. Discrete time vanilla FTRL    with entropic  regularizer (Euclidean projection algorithm)
5. Discrete time         FTRL+   with entropic  regularizer (exponential weights) in extra-gradient variant (not optimistic, two vector queries per step)

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
from aspera import utils
import time

# ------------------------------------------------
## PARAMETERS
# ------------------------------------------------

# ------------------------------------------------
## Grid density
# ------------------------------------------------
GRID_DENSITY =  6 # Number of initial conditions for continuous time trajectories, and points where payfield is quivered = square of this number

# If needed, fine-tune for side-to-side potential-harmonic plots
GRID_DENSITY_POTENTIAL =  GRID_DENSITY # Number of trajectories = square of this number; 9 is cool because grid coincides with contours intersections, but it's a bit too dense
GRID_DENSITY_HARMONIC = GRID_DENSITY  # Number of trajectories = square of this number
##################################################

# ------------------------------------------------
## Style
# ------------------------------------------------
PLAYER_1_COLOR = 'black'
PLAYER_2_COLOR = 'black'

REPLICATOR_COLOR = 'white'
PAYFIELD_COLOR = 'black'

VANILLA_COLOR = 'white'
EXTRA_COLOR = 'red'

EUCLIDEAN_COLOR = 'crimson'

POTENTIAL_FUNCTION_COLOR = 'red'

NE_COLOR = 'crimson'
INITIAL_POINT_COLOR = 'crimson'




# ------------------------------------------------
## Labels
# ------------------------------------------------
CONTINUOUS_TIME_LABEL = 'FTRL-D'
EXTRAPOLATION_LABEL = 'FTRL+'
VANILLA_LABEL = 'vanilla FTRL discrete'
PAYOFF_FIELD_LABEL = 'Payoff field'

ENTROPIC_LABEL = ' (entropic)'
EUCLIDEAN_LABEL = ' (euclidean)'

# NE_LABEL = "Strategic center"
NE_LABEL = "NE"

INCLUDE_LEGEND = 1
INCLUDE_AXES_LABELS = 1
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
ADD_COLORBAR = 0
DISPLAY_PAYOFF_CONTOURS_VALUES = 0 # tag each contour line with corresponding value

CONTOURS_DENSITY = 10 # number of contours lines
FILLED_CONTOUR_DENSITY = 100 # number of filled contour levels, higher = smoother shades transition

# ------------------------------------------------
## Quivers
# ------------------------------------------------
QUIVER_PAYFIELD = 1
QUIVER_INDIVIDUAL_PAYFIELD = 0
QUIVER_RD = 0

QUIVER_SCALE = 6 # Scaling for quiver plots; high number = short arrow RD is scaled by this number, payfield is scaled by this number SQUARED

# Finetune for potential and harmonic cases if needed
QUIVER_SCALE_POTENTIAL = QUIVER_SCALE # How much smaller potential arrows are then computed number
QUIVER_SCALE_HARMONIC = QUIVER_SCALE # How much smaller harmonic arrows are then computed number

PLOT_SEGMENT_PERPENDICULAR_HARMONIC_CENTER = 0


# ------------------------------------------------
## Continuous time dynamics
# ------------------------------------------------
SOLVE_ODE = 0
PLOT_CONTINUOUS_RD = 1
INDEX_1 = 1 #4 # choose orbit index, or plot them all
INDEX_2 = 3 # choose orbit index, or plot them all

PLOT_CONTINUOUS_EUCLIDEAN = 0

RD_LINEWIDTH = 1.5
ED_LINEWIDTH = 0.7

ODE_SOLVER_PRECISION = 1000 # high = more precise

CONTINUOUS_TIME_HORIZON_ENTROPIC = 20 # Max time for dynamical system ODE solver, replicator dynamics
CONTINUOUS_TIME_HORIZON_EUCLIDEAN = 0.1 # Max time for dynamical system ODE solver, Euclidean dynamics. Small bc usually hits boundary fast.

# ------------------------------------------------
## Discrete time dynamics
# ------------------------------------------------


# FEEDBACK_TYPE = "mixed_vector"                                                          # <---------------------------------------------------------- to check feedback types
# FEEDBACK_TYPE = "pure_vector"
FEEDBACK_TYPE = "bandit"
VANILLA_LABEL += f', {FEEDBACK_TYPE} feedback'

# entropic
PLOT_ENTROPIC_VANILLA_FTRL = 1
PLOT_EXTRA_FTRL = 0

EXTRA_FTRL_STEP_SIZE = 0.3
VANILLA_FTRL_STEPSIZE = 0.05
EXTRA_FTRL_LINEWIDTH = 1.5

# euclidean; not sure it's correct ! careful !
PLOT_EUCLIDEAN_VANILLA_FTRL = 0

TIMESTEPS_EXTRA_FTRL = 25000
TIMESTEPS_VANILLA_FTRL = 1000


# ------------------------------------------------
## Nash equilibria
# ------------------------------------------------
PLOT_NE = 1


# ------------------------------------------------
## Response graph
# ------------------------------------------------
DRAW_ORIENTED_EDGES = True # to draw arrow along edges of responde graph
ANNOTE_DEVIATION_VALUE = 1 # True to write on edges value of unilateral deviations
PURE_LABELS_CORNERS = 1 # Add label to pure strategies (which pure, payoff, potential if available)
PURE_LABELS_CORNERS_MIXED = 1 # Label mixed value of pure strategies

DEVIATION_FONT_SIZE = 30
PURES_FONT_SIZE = 20



####################################################################################################
############################################# BEGIN CORE ###########################################
####################################################################################################

class Game22():
    def __init__(self, payoff, metric, game_name, game_type = '', pure_potential_function = 0, strategies_labels = [ ['0', '1'], ['0', '1'] ]):

        '''
        - payoff is list with 8 numbers, flattenedpayoff matrices of two players
        - metric is string in ['eu', 'sha'], will be used to build continuous dynamics as sharp of reduced payoff field
        - game_name is string used in titles for plots and saved files
        - game type can be set to 'potential' or 'harmonic' to showcase side to side differences, but in general can be left empty
        - pure_potential_function is list with 4 numbers, if the game is potential it can be used to plot the potential contours
        - strategies_labels is a list of two lists with 2 strings each; default [ ['0', '1'], ['0', '1'] ], but each strategy can be given a name
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
        assert len(payoff) == 8
        self.payoff = payoff
        self.num_players = 2
        self.players = [i for i in range(1, self.num_players + 1)]

        '''
        Payoff bimatrix
        Access pure payoff u_i(a1, a2) as self.u_pure[i][a1][a2]

        for i in {1, 2}
        for a1 in {0, 1}
        for a2 in {0, 1}
        '''
        self.u_pure = { i : np.array(self.payoff[ (i-1)*4 : i*4 ]).reshape(2,2) for i in self.players  }

        # --------------------------------------------------------------------
        # coefficients of reduced payoff field, key
        # --------------------------------------------------------------------
        self.v1_lin_coeff =  ( + self.u_pure[1][0][0] + self.u_pure[1][1][1] - self.u_pure[1][1][0]  - self.u_pure[1][0][1]  )
        self.v1_aff_coeff =  ( - self.u_pure[1][0][0]                        + self.u_pure[1][1][0]                          )
        self.v2_lin_coeff = ( + self.u_pure[2][0][0] + self.u_pure[2][1][1] - self.u_pure[2][1][0]  - self.u_pure[2][0][1]  )
        self.v2_aff_coeff = ( - self.u_pure[2][0][0]                                                + self.u_pure[2][0][1]  )
        # --------------------------------------------------------------------
        self.interior_ne = self.find_fully_mixed_NE()

        self.strategies_labels = strategies_labels

        # Potential matrix
        if self.is_potential:
            try:
                assert pure_potential_function != 0
            except:
                raise Exception("For potential game need provide potential function")
            self.min_potential, self.max_potential = min(pure_potential_function), max(pure_potential_function)
            self.pure_potential_function = np.array(pure_potential_function).reshape(2,2)

    # --------------------------------------------------------------------
    ## Begin game learning methods
    # --------------------------------------------------------------------

    # --------------------------------------------------------------------
    ## Payoff bimatrix
    # --------------------------------------------------------------------
    def u(self, i, r1, r2):
        '''Reduced Mixed payoff. Here r1 = x11, r2 = x21 denote reduced variables from x = ( (x10, x11), (x20, x21) )
        Used to build reduced payfield, that in turn is used to give continuous time dynamcis via sharp of reduced metric.'''

        second_order =  r1 * r2 *   ( + self.u_pure[i][0][0] + self.u_pure[i][1][1] - self.u_pure[i][1][0]  - self.u_pure[i][0][1]  )
        first_order_1 = r1 *        ( - self.u_pure[i][0][0]                        + self.u_pure[i][1][0]                          )
        first_order_2 = r2 *        ( - self.u_pure[i][0][0]                                                + self.u_pure[i][0][1]  )
        zeroth_order =              ( + self.u_pure[i][0][0] )
        return second_order + first_order_1 + first_order_2 + zeroth_order


    # --------------------------------------------------------------------
    ## Reduced payfield
    # --------------------------------------------------------------------
    def v1(self, r1, r2):
        '''Reduced payoff field of player 1. Here r1 = x11, r2 = x21 denote reduced variables from x = ( (x10, x11), (x20, x21) )
        Note that v1 does not depend on r1: affine function

        v1(r1, r2) = a r2 + b

        Used to build continuous time dynamcis via sharp of reduced metric.
        Takes as input two scalars and returns one scalar.'''

        return self.v1_lin_coeff * r2 + self.v1_aff_coeff

    def v2(self, r1, r2):
        '''Reduced payoff field of player 2. Here r1 = x11, r2 = x21 denote reduced variables from x = ( (x10, x11), (x20, x21) )
        Note that v2 does not depend on r2: affine function

        v2(r1, r2) = a r1 + b

        Used to build continuous time dynamcis via sharp of reduced metric.
        Takes as input two scalars and returns one scalar.'''
         
        return self.v2_lin_coeff * r1 + self.v2_aff_coeff


    def payfield(self, r1, r2):
        # Reduced payfield: two scalars --> lenght-2 array
        # Here r1 = x11, r2 = x21 denote reduced variables from x = ( (x10, x11), (x20, x21) )
        return np.array([self.v1(r1, r2), self.v2(r1,r2)])


    # --------------------------------------------------------------------
    ## Full payfield
    # --------------------------------------------------------------------
    def full_payfield(self, x):
        '''Payoff field in full coordinates, flattened:  x = ( (x10, x11), (x20, x21) ) --> [ x10, x11, x20, x21 ].
        Necessary to work in full coordinates to get Euclidean projection on simplex for vanilla FTRL
        Takes as input and returns lenght-4 array'''

        x10, x11, x20, x21 = x # unpack length 4 array
        V10 = x20 * self.u_pure[1][0][0] + x21 * self.u_pure[1][0][1]
        V11 = x20 * self.u_pure[1][1][0] + x21 * self.u_pure[1][1][1]
        V20 = x10 * self.u_pure[2][0][0] + x11 * self.u_pure[2][1][0]
        V21 = x10 * self.u_pure[2][0][1] + x11 * self.u_pure[2][1][1]

        return np.array([V10, V11, V20, V21]) # return length 4 array


    # --------------------------------------------------------------------
    ## Reduced potential function
    # --------------------------------------------------------------------
    def mixed_potential_function(self, r1, r2):
        # Here r1 = x11, r2 = x21 denote reduced variables from x = ( (x10, x11), (x20, x21) )
        if not self.is_potential:
            return 0
        '''Reduced Mixed potential function'''
        second_order =  r1 * r2 *   ( + self.pure_potential_function[0][0] + self.pure_potential_function[1][1] - self.pure_potential_function[1][0]  - self.pure_potential_function[0][1]  )
        first_order_1 = r1 *        ( - self.pure_potential_function[0][0]                                      + self.pure_potential_function[1][0]                                        )
        first_order_2 = r2 *        ( - self.pure_potential_function[0][0]                                                                            + self.pure_potential_function[0][1]  )
        zeroth_order =              ( + self.pure_potential_function[0][0] )
        return second_order + first_order_1 + first_order_2 + zeroth_order


    # --------------------------------------------------------------------
    ## Reduced metrics: r1 = x11, r2 = x21 denote reduced variables from x = ( (x10, x11), (x20, x21) )
    # --------------------------------------------------------------------
    def sha_inv_1(self, r1, r2):
        # Reduced sha metric 1
        return r1 - r1**2

    def sha_inv_2(self, r1, r2):
        # Reduced sha metric 2
        return r2 - r2**2 # Sha


    # --------------------------------------------------------------------
    ## Reduced replicator dynamics continuous time: r1 = x11, r2 = x21 denote reduced variables from x = ( (x10, x11), (x20, x21) )
    # --------------------------------------------------------------------

    def RD1(self, r1, r2):
        return [self.sha_inv_1(r1, r2) * self.v1(r1, r2), 0]

    def RD2(self, r1, r2):
        return [0, self.sha_inv_2(r1, r2) * self.v2(r1, r2)]

    def RD(self, var, t):
        r1, r2 = var
        return [ self.RD1(r1, r2)[0], self.RD2(r1, r2)[1]  ]

    # --------------------------------------------------------------------
    ## Reduced euclidean dynamics continuous time # <------------------------------------------------------------- #TO-DO-EUCLIDEAN-CONTINUOUS-ANCHOR ------------------------------------------------------------------
    # --------------------------------------------------------------------

    def ED1(self, r1, r2):

        '''
        Euclidean metric sharp is identity # TO CHECK, NOT TRUE IN REDUCED COORDINATES
        r1 = x11, r2 = x21 denote reduced variables from x = ( (x10, x11), (x20, x21) )
        '''
        return [self.v1(r1, r2), 0]

    def ED2(self, r1, r2):
        return [0, self.v2(r1, r2)]

    def ED(self, var, t):
        r1, r2 = var
        return [ self.ED1(r1, r2)[0], self.ED2(r1, r2)[1]  ]

    def ED1_quiver(self, var, t):
        r1, r2 = var
        return [ self.ED1(r1, r2)[0], 0  ]

    def ED2_quiver(self, var, t):
        r1, r2 = var
        return [ 0, self.ED2(r1, r2)[1]  ]


    # --------------------------------------------------------------------
    ## FTRL+ (discrete time), entropic regularizer. Here x is REDUCED primal variable and y REDUCED dual variable. Extra-gradient variant (two gradient queries per step)
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
            x_lead = self.entropic_prox(x, self.payfield(*x), step)     # extrapolation step
            x = self.entropic_prox(x, self.payfield(*x_lead), step)     # main step
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

            # main step; take if out of for loop...

            if FEEDBACK_TYPE == "mixed_vector":
                oracle = self.payfield(*x)

            elif FEEDBACK_TYPE == "pure_vector":
                probabilities = [ [ 1-x1, x1 ] for x1 in x ]
                pure = np.array([ np.random.choice( [0,1], p = pi) for pi in probabilities ])
                oracle = self.payfield(*pure)

            elif FEEDBACK_TYPE == "bandit":
                # ugly implementation of importance weighted estimator, to check
                probabilities = [ [ 1-x1, x1 ] for x1 in x ]
                pure_1, pure_2 = np.array([ np.random.choice( [0,1], p = pi) for pi in probabilities ])
                pay = [ self.u_pure[i][pure_1][pure_2] for i in self.players ]

                IWE = [ [0,0], [0,0] ]
                IWE[0][pure_1] = pay[0] / probabilities[0][pure_1]
                IWE[1][pure_2] = pay[1] / probabilities[1][pure_2]
                oracle = np.array( [iwe[-1] for iwe in IWE ])




            else:
                raise Exception("Invalid feedback type")


            x = self.entropic_prox(x, oracle, step)


            trajectory.append(x)
            # step = step / t                  # update step size

        x1 = [x[0] for x in trajectory]
        x2 = [x[1] for x in trajectory]

        return x1, x2

    # --------------------------------------------------------------------
    ## Vanilla FTRL (discrete time), Euclidean regularizer. Here x is FULL primal variable and y FULL dual variable. Must work in full coordinates to project on simplex. <--------- #TO-DO-EUCLIDEAN-DISCRETE-ANCHOR --------------------------------------------------
    # --------------------------------------------------------------------

    def projection_simplex(self, v, z=1):
        # Full coordinates
        # v is lenght N array to be projected on (N-1) simplex; in this case N = 2
        # z I don't know, should be 1
        # taken from https://gist.github.com/mblondel/6f3b7aaad90606b98f71
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w

    def euclidean_prox(self, x, y, step):
        # Euclidean regularizer prox map, full coordinates
        # x and y are lenght 4 arrays
        # step is scalar

        x = np.array(x) # lenght 4 array
        y = np.array(y) # lenght 4 array
        p = x + step * y       # lenght 4 array, needs be projected on product of two 1-simplices

        p1, p2 = p[0:2], p[2:4]

        proj_1 = self.projection_simplex( p1 )
        proj_2 = self.projection_simplex( p2 )

        return np.concatenate( (proj_1, proj_2) ) # length 4 array 


    def vanilla_euclidean_ftrl(self, x0, num_iterations, initial_step):
        # Full coordinates
        x = np.array(x0) # size 4
        trajectory = [x0] # time series
        step = initial_step # scalar

        for t in range(1,num_iterations):
            x = self.euclidean_prox(x, self.full_payfield(x), step)     # main step
            trajectory.append(x)
            # step = step / t                  # update step size

        # extract first and third out of 
        x1 = [x[0] for x in trajectory]
        x2 = [x[2] for x in trajectory]

        return x1, x2

    # ------------------------------------------------
    ## NE finder
    # ------------------------------------------------



    def find_fully_mixed_NE(self):

        '''
        For interior NE to exist it must be slope, so must solve linear system for reduced payoff; two equations, two unknowns:

        v1(r1, r2) = 0
        v2(r1, r2) = 0

        v1(r1, r2) = a r2 + b = 0
        v2(r1, r2) = c r1 + d = 0

        r1 = - d/c
        r2 = - b/a

        The coefficients a, b, c, d depend on payoff bimatrix and are given in the definitions of v1 and v2

        '''

        a = self.v1_lin_coeff
        b = self.v1_aff_coeff
        c = self.v2_lin_coeff
        d = self.v2_aff_coeff

        ne_matrix = np.array( [ [0, a], [c, 0] ] )
        ne_affine_terms = np.array( [-b, -d] )

        try:
            ne = np.linalg.solve(ne_matrix, ne_affine_terms)
            print(f'Fully mixed NE: {ne}')
            return ne
        except:
            print ("No interior NE")
            return None
        

        


    # --------------------------------------------------------------------
    ## Begin plotting methods
    # --------------------------------------------------------------------

    def contour_plot(self, ax):

        # initialize empty legend elements to return, to avoid error in case this function is called but all the contours flags are False
        legend_elements = []

        # get max and min payoffs to set contours extreme values
        payoff_player_1, payoff_player_2 = self.payoff[0:4], self.payoff[4:8]
        min_payoff_player_1, max_payoff_player_1 = min(payoff_player_1),  max(payoff_player_1)
        min_payoff_player_2, max_payoff_player_2 = min(payoff_player_2),  max(payoff_player_2)

        UTILITY_LEVELS_1 = np.linspace(min_payoff_player_1, max_payoff_player_1, CONTOURS_DENSITY + 1)
        UTILITY_LEVELS_1_POTENTIAL = np.array([])
        UTILITY_LEVELS_1_HARMONIC = np.array([])

        # UTILITY_LEVELS_2 = np.linspace(min_payoff_player_2, max_payoff_player_2, CONTOURS_DENSITY + 1)

        # For specific example of Polaris seminar

        u_2_at_NE = self.u(2, 2/3, 1/2)

    
        first_half = np.linspace(min_payoff_player_2, u_2_at_NE, CONTOURS_DENSITY + 1)
        second_half = np.linspace(u_2_at_NE, max_payoff_player_2, CONTOURS_DENSITY + 1)[1:] 
     

        UTILITY_LEVELS_2 =  np.concatenate( (first_half, second_half) )


        UTILITY_LEVELS_2_POTENTIAL = UTILITY_LEVELS_1_POTENTIAL
        UTILITY_LEVELS_2_HARMONIC = UTILITY_LEVELS_1_HARMONIC + 0.5
        ##################################################


        if self.is_potential:
            utility_levels_1, utility_levels_2 = UTILITY_LEVELS_1_POTENTIAL, UTILITY_LEVELS_2_POTENTIAL
            contour_plot_levels_start = 1

        elif self.is_harmonic:
            utility_levels_1, utility_levels_2 = UTILITY_LEVELS_1_HARMONIC, UTILITY_LEVELS_2_HARMONIC
            contour_plot_levels_start = 1

        else:
            utility_levels_1, utility_levels_2 = UTILITY_LEVELS_1, UTILITY_LEVELS_2
            contour_plot_levels_start = 0

            
        y1 = np.linspace(0, 1, 100)
        y2 = np.linspace(0, 1, 100)
        Y1, Y2 = np.meshgrid(y1, y2)


        # label every n countour lines
        add_countour_label_every = 1


        if PLOT_CONTOURS_FIRST_PLAYER or PLOT_FILLED_CONTOURS_FIRST_PLAYER: data_1 = self.u(1, Y1, Y2 )

        # contours player 1
        if PLOT_CONTOURS_FIRST_PLAYER:
            contour_plot_1 = ax.contour(Y1, Y2, data_1, levels = utility_levels_1, colors = PLAYER_1_COLOR, linestyles = 'dashed', linewidths = 0.6)
            # legend_elements = [matplotlib.lines.Line2D([0], [0], color = PLAYER_1_COLOR, label = 'Payoff contours pl. 1',linestyle = 'dashed', linewidth= 0.6)]
            if DISPLAY_PAYOFF_CONTOURS_VALUES:
                [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in ax.clabel(contour_plot_1, levels = contour_plot_1.levels[contour_plot_levels_start::add_countour_label_every] ,  inline=True, fontsize=7, fmt='(%1.1f)')] # fmt rounds to one decimal place

        # filled contour player 1
        if PLOT_FILLED_CONTOURS_FIRST_PLAYER:

            # sometimes some error with levels = FILLED_CONTOUR_DENSITY * (max_payoff_player_1 - min_payoff_player_1)
            try:
                payoff_contourf = ax.contourf(Y1, Y2, data_1, cmap = 'viridis', alpha = 0.8, levels = FILLED_CONTOUR_DENSITY * (max_payoff_player_1 - min_payoff_player_1))
            except:
                payoff_contourf = ax.contourf(Y1, Y2, data_1, cmap = 'viridis', alpha = 0.8, levels = FILLED_CONTOUR_DENSITY )

            if PLOT_FILLED_CONTOURS_FIRST_PLAYER and ADD_COLORBAR:
                payoff_colorbar = plt.colorbar(payoff_contourf)
                payoff_colorbar.set_label(f'Payoff player 1 in [{min_payoff_player_1}, {max_payoff_player_1}] ', labelpad=20, rotation=90,  fontsize = 20) #, va='bottom')

        
        # contours player 2
        if PLOT_CONTOURS_SECOND_PLAYER:

            data_2 = self.u(2, Y1, Y2 ) 
            contour_plot_2 = ax.contour(Y1, Y2, data_2, levels = utility_levels_2, colors = PLAYER_2_COLOR, linestyles = 'dashdot', linewidths = 0.6)
            legend_elements.extend( [   matplotlib.lines.Line2D([0], [0], color = PLAYER_2_COLOR, label = 'Payoff contours pl. 2',linestyle = 'dashdot', linewidth = 0.6)] )

            if DISPLAY_PAYOFF_CONTOURS_VALUES:
                [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in ax.clabel(contour_plot_2, levels = contour_plot_2.levels[contour_plot_levels_start::add_countour_label_every] ,  inline=True, fontsize=7, fmt='(%1.1f)')]


        # contours potential function
        if self.is_potential and PLOT_CONTOURS_POTENTIAL_FUNCTION:
            data_potential = self.mixed_potential_function(Y1, Y2)
            UTILITY_LEVELS_POTENTIAL_FUNCTION = np.linspace(self.min_potential, self.max_potential, CONTOURS_DENSITY + 1)
            contour_plot_potential = ax.contour(Y1, Y2, data_potential, levels = UTILITY_LEVELS_POTENTIAL_FUNCTION, colors = POTENTIAL_FUNCTION_COLOR, linestyles = 'dashdot', linewidths = 0.6)
            if DISPLAY_PAYOFF_CONTOURS_VALUES:
                [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0)) for txt in ax.clabel(contour_plot_potential, levels = contour_plot_potential.levels[0::add_countour_label_every] ,  inline=True, fontsize=7, fmt='(%1.1f)' )]
            legend_elements.extend( [ matplotlib.lines.Line2D([0], [0], color = POTENTIAL_FUNCTION_COLOR, label = 'Potential function',linestyle = 'dashdot', linewidth = 0.6) ] )

        return legend_elements

        

    def quiver_RD_plot(self, ax):

        if self.is_potential:
            y1_range, y2_range = np.linspace(0, 1, GRID_DENSITY_POTENTIAL), np.linspace(0, 1, GRID_DENSITY_POTENTIAL)

            # exclude boundaries
            # y1_range, y2_range = y1_range[1:-1], y2_range[1:-1]

        elif self.is_harmonic:
            y1_range, y2_range = np.linspace(0, 1, GRID_DENSITY_HARMONIC), np.linspace(0, 1, GRID_DENSITY_HARMONIC)

        else:
            y1_range, y2_range = np.linspace(0, 1, GRID_DENSITY), np.linspace(0, 1, GRID_DENSITY)


        y1 = y1_range
        y2 = y2_range


        Y1, Y2 = np.meshgrid(y1, y2)

        # Replicator dynamics, continuous time
        RD = self.RD((Y1, Y2), 0)


        # Euclidean dynamics, continuous time
        ED = self.ED((Y1, Y2), 0)

        
        if self.is_potential:
            scale = QUIVER_SCALE_POTENTIAL
        elif self.is_harmonic:
            scale = QUIVER_SCALE_HARMONIC
        else:
            scale = QUIVER_SCALE

        # Quiver for each player
        # ax.quiver(Y1, Y2, *RD1, scale=2.5, color = PLAYER_1_COLOR, width=0.005, headlength=5)
        # ax.quiver(Y1, Y2, *RD2, scale=2.5, color = PLAYER_2_COLOR, width=0.005, headlength=5)

        # Quiver replicator field
        if QUIVER_RD:
            ax.quiver(Y1, Y2, *RD, scale = scale, color = REPLICATOR_COLOR, width=0.003, headlength=3)
        
        # Quiver payoff field
        if QUIVER_PAYFIELD:
            ax.quiver(Y1, Y2, *ED, scale = scale * scale, color =  PAYFIELD_COLOR, width=0.005, headlength=4)

        if QUIVER_PAYFIELD and QUIVER_INDIVIDUAL_PAYFIELD:

            ED1 = self.ED1_quiver((Y1, Y2), 0)
            ED2 = self.ED2_quiver((Y1, Y2), 0)

            ax.quiver(Y1, Y2, *ED1, scale = scale * scale, color = PLAYER_1_COLOR, width=0.003, headlength=3)
            ax.quiver(Y1, Y2, *ED2, scale = scale * scale, color = PLAYER_2_COLOR, width=0.003, headlength=3)



        # legend_elements = [ ]

        #if self.is_potential:
        # payoff_field_label = 'Payoff field'
        #payoff_field_label = 'Payoff'
        # legend_elements = [ ]
        legend_elements =  [matplotlib.lines.Line2D([0], [0], color= PAYFIELD_COLOR, label = PAYOFF_FIELD_LABEL)] 


        # extra gradient entropic ftrl
        if PLOT_EXTRA_FTRL:

            # pick an initial point
            initial_point = np.array([ y1[1], y2[3] ])
            print(f'Initial point of extra ftrl: {initial_point}')
            plt.scatter(*initial_point, color = INITIAL_POINT_COLOR, s = 20)
            EGMD = self.extra_ftrl( initial_point, TIMESTEPS_EXTRA_FTRL, EXTRA_FTRL_STEP_SIZE )
            nash_x, nash_y = EGMD[0][-1], EGMD[1][-1]

            plt.plot(*EGMD, color = EXTRA_COLOR,  linewidth = EXTRA_FTRL_LINEWIDTH, label = EXTRAPOLATION_LABEL + ENTROPIC_LABEL)
            # plt.scatter( [nash_x], [nash_y], color = 'black', s = 10 )
            legend_elements.extend( [matplotlib.lines.Line2D([0], [0], color = EXTRA_COLOR, label=EXTRAPOLATION_LABEL + ENTROPIC_LABEL)]  )

        # vanilla entropic ftrl
        if PLOT_ENTROPIC_VANILLA_FTRL:
            # initial_point = np.array([ y1[1], y2[3] ])
            initial_point = np.random.rand(2)
            print(f'Initial point of entropic vanilla ftrl: {initial_point}')
            plt.scatter(*initial_point, color = INITIAL_POINT_COLOR, s = 20)
            vanilla_entropic_ftrl_trajectory = self.vanilla_entropic_ftrl( initial_point, TIMESTEPS_VANILLA_FTRL, VANILLA_FTRL_STEPSIZE )
            plt.plot(*vanilla_entropic_ftrl_trajectory, color = VANILLA_COLOR, linewidth = 0.7, label = VANILLA_LABEL + ENTROPIC_LABEL)
            legend_elements.extend( [matplotlib.lines.Line2D([0], [0], color = VANILLA_COLOR, label = VANILLA_LABEL  + ENTROPIC_LABEL)]  )


         # vanilla euclidean ftrl (to check)
        if PLOT_EUCLIDEAN_VANILLA_FTRL:
            # pick an initial point
            initial_point = np.array([ y1[1], 1-y1[1], y2[3], 1-y2[3] ])
            print(f'Initial point of euclidean vanilla ftrl: {initial_point}')
            plt.scatter(*initial_point, color = INITIAL_POINT_COLOR, s = 20)
            print(initial_point)
            plt.scatter(initial_point[0], initial_point[2], color = 'black')
            EUCLIDEAN_MIRROR_DESCENT = self.vanilla_euclidean_ftrl( initial_point, TIMESTEPS_VANILLA_FTRL, 0.01 )
            plt.plot(*EUCLIDEAN_MIRROR_DESCENT, color = EUCLIDEAN_COLOR,  linewidth = 0.7, label = VANILLA_LABEL + EUCLIDEAN_LABEL)
            legend_elements.extend( [matplotlib.lines.Line2D([0], [0], color = EUCLIDEAN_COLOR, label = VANILLA_LABEL + EUCLIDEAN_LABEL )]  )

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

        # Replicator
        if PLOT_CONTINUOUS_RD:
            ax.plot(*zip(*odeint(self.RD, start[0], t_sha)), color = REPLICATOR_COLOR, linewidth = RD_LINEWIDTH)
            [ ax.plot(*zip(*odeint(self.RD, p, t_sha)), color = REPLICATOR_COLOR, linewidth = RD_LINEWIDTH) for p in start[1:] ]
            legend_elements.extend( [ matplotlib.lines.Line2D([0], [0], color = REPLICATOR_COLOR, label = CONTINUOUS_TIME_LABEL + ENTROPIC_LABEL ) ])

        # Euclidean
        if PLOT_CONTINUOUS_EUCLIDEAN:
            ax.plot(*zip(*odeint(self.ED, start[0], t_eu)), color = PAYFIELD_COLOR, linewidth = ED_LINEWIDTH, linestyle = '--')
            [ ax.plot(*zip(*odeint(self.ED, p, t_eu)), color = PAYFIELD_COLOR, linewidth = ED_LINEWIDTH, linestyle = '--') for p in start[1:] ]
            legend_elements.extend( [matplotlib.lines.Line2D([0], [0], color = PAYFIELD_COLOR, label = CONTINUOUS_TIME_LABEL + EUCLIDEAN_LABEL, linestyle = '--', linewidth = ED_LINEWIDTH)] )

        return  legend_elements

    def full_plot(self,ax):

        legend_elements = []

        if PLOT_NE:
            try:
                plt.scatter(self.interior_ne[0], self.interior_ne[1], color = NE_COLOR, zorder=10, s = 60) # zorder is like z-index in css, higher plots this point on top of other graphical elements. Need high else the contourfill and the extra FTRL cover it
                legend_elements.extend( [matplotlib.lines.Line2D([0], [0], color = NE_COLOR, label = NE_LABEL, linestyle = '', marker = 'o' )] )

            except:
                pass

        if PLOT_CONTOURS:
            legend_elements.extend(self.contour_plot(ax))

        if SOLVE_ODE:
            legend_elements.extend(self.ode_plot(ax))

        if QUIVER_PAYFIELD:
            legend_elements_quiver, GRID_1, GRID_2 = self.quiver_RD_plot(ax)
            legend_elements.extend(legend_elements_quiver)


        if INCLUDE_AXES_LABELS:
            ax.set_xlabel(f'Prob. player 1 assigns to {self.strategies_labels[0][1]} in {{{self.strategies_labels[0][0]}, {self.strategies_labels[0][1]}}}', fontsize=AXES_LABEL_FONT_SIZE)
            ax.set_ylabel(f'Prob. player 2 assigns to {self.strategies_labels[1][1]} in {{{self.strategies_labels[1][0]}, {self.strategies_labels[1][1]}}}', fontsize=AXES_LABEL_FONT_SIZE)


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


        if INCLUDE_LEGEND:
            ax.legend(handles=legend_elements, loc='upper left', fontsize = 10)



        plt.xlim(0,1)
        plt.ylim(0,1)
        ax.set_aspect('equal', adjustable='box')


        # -------------------------------------------------------------------------
        ## Response graph
        # -------------------------------------------------------------------------

        # -------------------------------------------------------------------------
        ## Label pures
        # -------------------------------------------------------------------------
        # Get the limits of the plot
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        [ [a10, a11], [a20, a21]] = self.strategies_labels

        # Here manage position of pures labels
        shift = 0.05

        # Add labels to the corners
        if PURE_LABELS_CORNERS:

            if PURE_LABELS_CORNERS_MIXED:
                mixed_top_left = "= (0,1)"
                mixed_top_right = "= (1,1)"
                mixed_bottom_left = "= (0,0)"
                mixed_bottom_right = "= (1,0)"

            else:
                mixed_top_left = ""
                mixed_top_right = ""
                mixed_bottom_left = ""
                mixed_bottom_right = ""
                # remove mixed values along axes
                plt.gca().set_xticks([])
                plt.gca().set_yticks([])

            if self.is_potential:
                pot_top_left = f", $\\phi$ = {self.pure_potential_function[0][1]}"
                pot_top_right = f", $\\phi$ = {self.pure_potential_function[1][1]}"
                pot_bottom_left = f", $\\phi$ = {self.pure_potential_function[0][0]}"
                pot_bottom_right = f", $\\phi$ = {self.pure_potential_function[1][0]}"
            else:
                pot_top_left = ""
                pot_top_right = ""
                pot_bottom_left = ""
                pot_bottom_right = ""




            # do include mixed values of pure strategies
            plt.text(x_min - 3 * shift, y_max + 1.5 * shift,   f'({a10},{a21}) {mixed_top_left}: u = ({self.u_pure[1][0][1]}, {self.u_pure[2][0][1]}){pot_top_left}', verticalalignment='top',    horizontalalignment='left',    fontsize = PURES_FONT_SIZE)
            plt.text(x_max + 3 * shift, y_max + 1.5 * shift,   f'({a11},{a21}) {mixed_top_right}: u = ({self.u_pure[1][1][1]}, {self.u_pure[2][1][1]}){pot_top_right}', verticalalignment='top',    horizontalalignment='right',   fontsize = PURES_FONT_SIZE)
            plt.text(x_min - 3 * shift, y_min - 1.5 * shift,   f'({a10},{a20}) {mixed_bottom_left}: u = ({self.u_pure[1][0][0]}, {self.u_pure[2][0][0]}){pot_bottom_left}', verticalalignment='bottom', horizontalalignment='left',     fontsize = PURES_FONT_SIZE)
            plt.text(x_max + 3 * shift, y_min - 1.5 * shift,   f'({a11},{a20}) {mixed_bottom_right}: u = ({self.u_pure[1][1][0]}, {self.u_pure[2][1][0]}){pot_bottom_right}', verticalalignment='bottom', horizontalalignment='right',    fontsize = PURES_FONT_SIZE)


                
            
            



       

        # -------------------------------------------------------------------------
        ## Orient edges
        # -------------------------------------------------------------------------

        # Count # of incoming flows to find pure NE

        n_in_top_right_strict = 0
        n_in_top_left_strict = 0
        n_in_bottom_right_strict = 0
        n_in_bottom_left_strict = 0

        n_in_top_right_loose = 0
        n_in_top_left_loose = 0
        n_in_bottom_right_loose = 0
        n_in_bottom_left_loose = 0

        # Response graph edges arrow style

        arr_style = ",head_length=1, head_width=1"

        # -------------------------------------------------------------------------
        # Bottom border
        deviation = self.u_pure[1][1][0] - self.u_pure[1][0][0]
        if deviation > 0:
            n_in_bottom_right_strict += 1
            arr = "->"
        elif deviation < 0:
            n_in_bottom_left_strict += 1
            arr = "<-"
        else:
            n_in_bottom_right_loose += 1
            n_in_bottom_left_loose += 1
            arr = "<->"
        # arrow
        if DRAW_ORIENTED_EDGES: plt.annotate('', xy=(x_max, y_min), xytext=(x_min, y_min), arrowprops=dict(arrowstyle = arr + arr_style, lw=3))
        # deviation value
        # if ANNOTE_DEVIATION_VALUE: plt.text(x_max / 2, y_min-5*shift, f"bottom = {abs(deviation)}", color = 'red', fontsize = DEVIATION_FONT_SIZE)
        if ANNOTE_DEVIATION_VALUE: plt.text(x_max / 2, y_min-2*shift, f"{abs(deviation)}", color = 'red', fontsize = DEVIATION_FONT_SIZE)

        # -------------------------------------------------------------------------
        # Top border
        deviation = self.u_pure[1][1][1] - self.u_pure[1][0][1]
        if deviation > 0:
            n_in_top_right_strict += 1
            arr = "->"
        elif deviation < 0:
            n_in_top_left_strict += 1
            arr = "<-"
        else:
            n_in_top_right_loose += 1
            n_in_top_left_loose += 1
            arr = "<->"
        # arrow
        if DRAW_ORIENTED_EDGES: plt.annotate('', xy=(x_max, y_max), xytext=(x_min, y_max),arrowprops=dict(arrowstyle = arr + arr_style, lw=3))
        # deviation value
        #if ANNOTE_DEVIATION_VALUE: plt.text(x_max / 2, y_max + 1.5*shift, f"top = {abs(deviation)}", color = 'red', fontsize = DEVIATION_FONT_SIZE)
        if ANNOTE_DEVIATION_VALUE: plt.text(x_max / 2, y_max + 1.5*shift, f"{abs(deviation)}", color = 'red', fontsize = DEVIATION_FONT_SIZE)

        # -------------------------------------------------------------------------
        # Right border
        deviation = self.u_pure[2][1][1] - self.u_pure[2][1][0]
        if deviation > 0:
            arr = "->"
            n_in_top_right_strict += 1
        elif deviation < 0:
            n_in_bottom_right_strict += 1
            arr = "<-"
        else:
            n_in_top_right_loose += 1
            n_in_bottom_right_loose += 1
            arr = "<->"
        # arrow
        if DRAW_ORIENTED_EDGES: plt.annotate('', xy=(x_max, y_max), xytext=(x_max, y_min),arrowprops=dict(arrowstyle = arr + arr_style, lw=3))

        # deviation value

        # with text
        # if ANNOTE_DEVIATION_VALUE: plt.text(x_max + 1.5*shift , y_max / 2, f"right = {abs(deviation)}", color = 'red', fontsize = DEVIATION_FONT_SIZE)

        # only number
        if ANNOTE_DEVIATION_VALUE: plt.text(x_max + 1.5*shift , y_max / 2, abs(deviation ), color = 'red', fontsize = DEVIATION_FONT_SIZE)

        # Fix number of digits
        # fixed_digits_deviation = f"{abs(deviation):.3f}"
        # if ANNOTE_DEVIATION_VALUE: plt.text(x_max + 1.5*shift , y_max / 2, fixed_digits_deviation, color = 'red', fontsize = DEVIATION_FONT_SIZE)


        # -------------------------------------------------------------------------
        # Left border
        deviation = self.u_pure[2][0][1] - self.u_pure[2][0][0]
        if deviation > 0:
            arr = "->"
            n_in_top_left_strict += 1
        elif deviation < 0:
            n_in_bottom_left_strict += 1
            arr = "<-"
        else:
            n_in_top_left_loose += 1
            n_in_bottom_left_loose += 1
            arr = "<->"
        # arrow
        if DRAW_ORIENTED_EDGES: plt.annotate('', xy=(x_min, y_max), xytext=(x_min, y_min), arrowprops=dict(arrowstyle = arr + arr_style, lw=3))
        # deviation value
        # if ANNOTE_DEVIATION_VALUE: plt.text(x_min - 11*shift, y_max / 2, f"left = {abs(deviation)}", color = 'red', fontsize = DEVIATION_FONT_SIZE)
        if ANNOTE_DEVIATION_VALUE: plt.text(x_min - 3*shift, y_max / 2, f"{abs(deviation)}", color = 'red', fontsize = DEVIATION_FONT_SIZE)


        # -------------------------------------------------------------------------
        ## Pure NE
        # -------------------------------------------------------------------------

        # number of strictly incoming flows per vertex
        n_in_strict = np.array([ n_in_bottom_left_strict, n_in_top_left_strict, n_in_top_right_strict, n_in_bottom_right_strict ])

        # number of loosely incoming flows per vertex
        n_in_loose = np.array([ n_in_bottom_left_loose, n_in_top_left_loose, n_in_top_right_loose, n_in_bottom_right_loose ])

        # number of incoming flows per vertex (either strict or loose)
        n_in = n_in_strict + n_in_loose

        
        # clockwise, starting from (0,0) = bottom left
        pures = [  [0,0], [0,1], [1,1], [1,0]  ]

        # vertex is strict NE if has 2 strictly incoming flows, and loose NE if has 2 incoming flows

        # select vertices with 2 strictly incoming flows
        self.pure_strict_NE =    [ pures[a] for a in range(len(pures)) if n_in_strict[a] == 2 ]

        # select vertices with 2 incoming flows, excluding the ones already counted as strict
        self.pure_loose_NE =     [ pures[a] for a in range(len(pures)) if n_in[a] == 2 and pures[a] not in self.pure_strict_NE ]

        print(f'Pure non-strict NE: {self.pure_loose_NE}')
        print(f'Pure strict NE: {self.pure_strict_NE}')

        if PLOT_NE:
            for ne in self.pure_strict_NE:
                plt.scatter(ne[0], ne[1], color = NE_COLOR, zorder=10, s = 100)

            for ne in self.pure_loose_NE:
                plt.scatter(ne[0], ne[1], edgecolor = NE_COLOR, facecolor = 'none', zorder=10, s = 400, linewidths = 2)


        # Hard coded removed axes ticks labels
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

        # -------------------------------------------------------------------------
        ## Harmonic: print "perpendicular" vector from center
        # -------------------------------------------------------------------------
        if PLOT_SEGMENT_PERPENDICULAR_HARMONIC_CENTER:
            grid_index_1 = [INDEX_1]
            grid_index_2 = [INDEX_2]

            try:
                for i in range(len(grid_index_1)):
                    plt.plot([self.interior_ne[0], GRID_1[ grid_index_1[i] ]], [self.interior_ne[1], GRID_2[ grid_index_2[i] ]], 'k--' )
            except:
                print("You're trying to plot harmonic center, but there is no interior Nash. Switch PLOT_SEGMENT_PERPENDICULAR_HARMONIC_CENTER to False")

    # ---------------------------------------
    def return_NE_info(self):
        return f"""
        Fully mixed NE: {self.interior_ne}\n
        Pure non-strict NE: {self.pure_loose_NE}\n
        Pure strict NE: {self.pure_strict_NE}\n
        """


##################################################################################################

# --------------------------------------------------------
## Bestiario di giochi
# --------------------------------------------------------

################ RANDOM, COOL ################
# payoff_potental = [-17, -4, -7, 4, -8, 7, -8, 5]
# payoff_harmonic = [-21, 12, -10, 1, 3, -8, -9, 2]
# G_pot = Game22(payoff_potental, 'sha', 'Potential')
# G_harm = Game22(payoff_harmonic, 'sha', 'Harmonic')
# G_pot.full_plot(axs[0])
# G_harm.full_plot(axs[1])
###############################################

################ Prisoner Dilemma and Matching Pennies ################
# payoff_PD = [2, 0, 3, 1, 2, 3, 0, 1]        # prisoner's dilemma, potential
# payoff_MP = [3, -3, -3, 3, -3, 3, 3, -3]    # matching pennies, harmonic
# G_PD = Game22(payoff_PD, 'sha', 'potential',  '''Prisoner's Dilemma''', pure_potential_function = [-1, 0, 0, 1])
# G_MP = Game22(payoff_MP, 'sha', 'harmonic',  'Matching pennies')

# fig, axs = plt.subplots(1, 2, figsize=(10, 4))#, dpi = 2400)
# G_PD.full_plot(axs[0])
# G_MP.full_plot(axs[1])
###############################################

################ EXPLORE ################
# payoff_explore = 5 * np.array([1, 1, 0, 0, 0, 1, 0, 0]) # Good to see how Sha gradient dapends towards boundary, while Eu gradient gets out of simplex
# G_sha = Game22(payoff_explore, 'sha', 'Explore')
# G_eu = Game22(payoff_explore, 'eu', 'Explore')
# G_sha.full_plot(axs[0])
# G_eu.full_plot(axs[1])
###############################################


################ RANDOM ################
# G_rd_1 = Game22(np.random.randint(-5, 5, 8), 'Random')
# G_rd_2 = Game22(np.random.randint(-5, 5, 8), 'Random')
# G_rd_1.full_plot(axs[0])
# G_rd_2.full_plot(axs[1])
###############################################

################ RANDOM ################
# G_rd_1 = Game22(np.random.randint(-5, 5, 8), 'Random')
# G_rd_2 = Game22(np.random.randint(-5, 5, 8), 'Random')
# G_rd_1.full_plot(axs[0])
# G_rd_2.full_plot(axs[1])
###############################################

# Bestiario di giochi

# generalized harmonic
# payoff = [-1, -4, -3, -1, 1, 2, 2, -2] 

# mathing pennies
# payoff = [3, -3, -3, 3, -3, 3, 3, -3]

# prisoner's dilemma
# payoff = [2, 0, 3, 1, 2, 3, 0, 1]

# split or steal
# payoff = [5, 0, 10, 0, 5, 10, 0, 0]

# battle of sexes
# payoff = [3, 0, 0, 2, 2, 0, 0, 3]

# chicken
# payoff = [-5, 2, 1, 0, -5, 1, 2, 0]

# --------------------------------
# https://arxiv.org/pdf/1701.09043
# --------------------------------

# coordination
payoff = [5, 1, 1, 4, 5, 1, 1, 4]

# anti-coordination
# payoff = [1, 5, 4, 1, 1, 4, 5, 1]

# cyclic
# payoff = [5, 1, 1, 4, -5, 1, 1, -4]

# dominance-solvable
# payoff = [1, 3, 0, 2, 1, 0, 3, 2]

# celia
# payoff = [4, 9, 9, 5, 0, 1, 12, 3 ]
###############################################

# --------------------------------------------------
# POLARIS JUNE 2024: 2x2 generalized harmonic gamee
# --------------------------------------------------


# --------------------------------------------------
# CRUCIAL: Harmonic center
# --------------------------------------------------
# These 4 numbers are free
# Changing them chages the NE of the game, given by MU / |MU|
# Equivalently, changes inner product underpinning notion of harmonicity

# Candogan harmonic:
MU = [ [ 0, 1 ], [ 1, 1 ]  ]

# MU = [ [ 1, 2 ], [ 1, 1 ]  ]

# --------------------------------------------------
# Harmonic deviations
# --------------------------------------------------
# For a given harmonic measure / center, build deviations (flows, differences between unilateral deviations) fulfilling the harmonic condition
# One edge is free (left), the other 3 are constrained (right, top, bottom)

# Changing the deviation for fixed measure amounts to non-strategic changes:
# - Unchanged dynamics and NE
# - Different payoff contours
# - Can tune them to make graph look nice

# -----------------------------
# Standard Matching pennies
left = 2
# -----------------------------
# left = 1
# -----------------------------
# Constrained deviations
right = MU[0][0] / MU[0][1] * left
top = MU[1][0] / MU[0][1] * left
bottom = MU[1][1] / MU[0][1] * left
####################################

# --------------------------------------------------
# Harmonic Payoff
# --------------------------------------------------
# Payoff realizing such deviations

# These 4 numbers are free, and changing them amounts to to non-strategic changes:
# - Unchanged dynamics and NE
# - Different payoff contours
# - Can tune them to make graph look nice

# -----------------------------
# Standard Matching pennies
# alpha = +1
# beta = -1
# gamma = -1
# delta = -1
# -----------------------------
alpha = +1
beta = -1
gamma = -1
delta = -1
# -----------------------------

# HARMONIC PAYOFF
# payoff = [alpha, gamma, alpha-bottom, gamma+top, beta, beta+left, delta+right, delta]

# harmonic payoff, no commas
# payoff = [1, -1, 0, 0, -1, 1, 0, -1]

# End POLARIS June 2024
###############################################



# --------------------------------------------------
# Potential 2x2 game
# --------------------------------------------------

"""potential function, 4 dofs"""
pot_AA = 0
pot_AB = 1
pot_BA = 1
pot_BB = 2

# Feed to game instance : pure_potential_function = [pot_AA, pot_AB, pot_BA, pot_BB]

"""deviations fulfilling potential constraint, all fixed"""
pot_left = pot_AA - pot_AB
pot_right = pot_BB - pot_BA
pot_top = pot_BB - pot_AB
pot_bottom = pot_AA - pot_BA

"""payoff realizing such deviations, 4 dofs (alpha, beta, gamma, delta); 4 constrained"""
# payoff = [alpha, delta, alpha - pot_bottom, delta + pot_top, beta, beta - pot_left , gamma, gamma + pot_right]
G = Game22(payoff, 'sha', game_type = '',  game_name = 'coordination_oracle_feedback', pure_potential_function = [pot_AA, pot_AB, pot_BA, pot_BB], strategies_labels = [ ['A', 'B'], ['C', 'D'] ] )
print(G.payoff)
# --------------------------------------------------
# End potential 2x2 game
# --------------------------------------------------


# --------------------------------------------------------
## Game instance
# --------------------------------------------------------
# assedio
# payoff = [-3, 2, 0, 0, 1, -4, -1, 0]
# G = Game22(payoff, 'sha', game_type = '',  game_name = 'extra_ftrl_22_siege_thumbnail', pure_potential_function = [pot_AA, pot_AB, pot_BA, pot_BB], strategies_labels = [ ['A', 'N'], ['D', 'N'] ] )



# boundary garmonic
# u = [9, -7, -1, -2, -2, -2, 2, -3]
# payoff = [-6, 9, 4, 4, 2, 2, -4, 1]


# Potential with pure non strict NE 2x2
# u = [0, 3, 1, 3, 0, 1, 2, 2]
# G = Game22(u, 'sha', game_type = 'potential',  game_name = 'potential pure non strict NE', strategies_labels = [ ['L', 'R'], ['B', 'T'] ], pure_potential_function = [0, 1, 1, 1] )

# u =  [0, 1, 5, 6, 6, 1, 5, 0]
# G = Game22(u, 'sha', game_type = '',  game_name = 'pot', strategies_labels = [ ['L', 'R'], ['B', 'T'] ] )



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
###############################################


# --------------------------------------------------------
## Save methods
# --------------------------------------------------------

root = './Results/'

game_directory = f'{root}/{G.game_name}'
current_directory = f'{game_directory}/{time.time()}'

utils.make_folder(current_directory)
plt.savefig(f'{current_directory}/{G.game_name}.pdf', bbox_inches='tight')#, pad_inches = 0)
utils.write_to_txt(f'{current_directory}/{G.game_name}.txt', G.u)
utils.write_to_txt(f'{current_directory}/{G.game_name}.txt', G.return_NE_info())


#####################################################


plt.show()






