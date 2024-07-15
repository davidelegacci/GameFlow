"CAREFUL WITH GLOBAL MINUS SIGN"

"DO NOT USE FOR EXPERIMENTS -  cool for theory but not efficient because it works in full coordinates"

"Work in full coordinates, cumbersome to pack and flatten x. This version is ok for theoretical playing around
because the notation is like writing maths by hand; for performance much better to use the reduced version"

"Plots 3 trajectories:
- continuous time RD
- Vanilla mirror descent with entropic distance generating function, i.e. exponential weights
- [DEPRECATED] Euler discretization of RD in primal space x --> x + step * RD(x)

y --> y + step * v(x)
x = Q(y)
Q choice or mirror map

equivalent to

x --> P(x,y)
P prox map
"

# N player RD is ready and ok; last issue: compatibility of payoff input ordering from Candogan code


println("\n START")

# to save figure in current folder
cd(dirname(@__FILE__()))

# using StrategicGames
using DifferentialEquations
using Plots
using IterTools
using StrategicGames
using Distances



"""
Implementattion of 3 players  [2,2,2] finite game in normal form and replicator dynamics
"""

# Number of actions for each player
A = [2,2,3]
nPures = prod(A)

skeleton = A # alias
@info "-----------BEGIN------------"
@info "Skeleton: $skeleton"

#  Number of players
N = length(A)
players = 1:N

# Pure strategies


pures_play = [ 1:A[i] for i in players ]


# pures in 2x2x2
# pures = [(a1, a2, a3) for a1 in pures_play[1], a2 in pures_play[2], a3 in pures_play[3]]

# pures in N player

# wrong ordering; needs permutation
# pures = collect(product((1:ai for ai in skeleton)...))

# ordering compatible with candogan
ranges = (1:ai for ai in skeleton)
reversed_ranges = reverse(collect(ranges))
pures_to_order = collect(product(reversed_ranges...))
pures = [reverse(a) for a in pures_to_order]
for a in pures
    println(a)
end





# RANDOM payoff and initial

# random payoff tensor as A_1 x A_2 x ... x A_N x N tensor, with values between -5 and 5 #------------------------------------------------------------------------- Payoff and initial point
# u_pure = rand(-5:5, A..., N)

#-------------------------------------------------------------------------
# Random initial condition

INITIAL_POINTS = [  ]

for _ in 1:10
    x0_ambient = [ rand(Ai) for Ai in skeleton  ]
    x0 = [ xi ./sum(xi) for xi in x0_ambient ]
    push!(INITIAL_POINTS, x0)
end



# Global initial point
x_start_ambient = [ rand(Ai) for Ai in skeleton  ]
x_start = [ xi ./sum(xi) for xi in x_start_ambient ]


#-------------------------------------------------------------------------

# payoff 2223
# payoff = [-40, 19, -16, 23, -5, -10, -29, -2, 20, 4, -7, 1, 0, 4, -3, 1, 1, -2, -3, -1, 3, 2, 0, -4, -83, 29, 19, 16, -25, 12, 0, -2, -2, 1, -3, 0, -3, -3, 3, 1, 1, 0, 1, 0, 2, -5, 3, 3, 99, -103, 29, 3, -2, -4, -5, 4, -4, 1, -3, -1, -4, 3, -3, -3, 0, 1, -2, -4, 1, 2, -5, -1, 21, 77, 2, 0, -5, 1, 2, 4, 3, -4, -5, -5, -4, 2, 0, -4, -4, -5, -3, -2, 1, 3, 1, 3]

# 223 generalized harmonic
payoff = [-4, -38, 27, -7, -33, 26, -2, 2, 4, 2, -2, -2, -7, -50, 45, 2, -5, 4, 2, 0, 0, 4, 4, -1, 26, 49, 1, 3, 4, 0, 2, -2, 4, -2, -5, 3]

# x_start = [[0.6542654903782231, 0.3457345096217769], [0.669321063305893, 0.3306789366941069], [0.20217140123604116, 0.35865642484259164, 0.43917217392136715]]

# HARD CODED payoff and initial
# Reshape and permutation for compatibility with ordering of payoff output by Candogan code, generate_harmonic.py, mixed_extension.py, etc. in 2x2x2 case.
# Swap first and third strategy index, invariant second strategy index and player index, achieved by permutedims(... , (3, 2, 1, 4))

# payoff = [-4, -3, 33, -29, 1, -4, 9, -9, 1, -8, -10, -1, 0, 7, 9, 2, -12, -6, -10, 3, 3, -1, 5, -10]
# payoff = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# payoff = [1, 2, 3, 4, 5, 6, 7, 8]

# payoff = [3, -2, -5, 14, 2, -9, 2, -4, 4, 11, -10, 3, 3, 1, 1, 1, -2, 0, 3, 2, 3, 0, -2, -1, 1, -10, 1, 13, -6, 4, -2, -2, 3, 0, 3, 0, 2, -3, 0, 2, -2, 3, 1, -2, 3, -2, 1, 2, 26, -2, -5, 3, -3, 3, 3, 0, -1, 1, 1, 1, -3, -1, 1, 0, 3, 0, -3, 1, -2, 3, -1, 3, -11, 1, 3, -1, 2, -1, 3, 2, 2, 1, -1, -1, 3, 3, 1, 2, 1, -3, 3, -2, 2, 3, -3, 0]

# HARMONIC 222 IN CANDOGAN CODE ORDER
# payoff = [-4, 8, 6, -2, 1, 2, 2, 3, 6, -6, 0, 1, -2, 1, 1, -3, -3, -2, -1, -3, 0, 2, -2, -3]
# payoff = [0, -2, 2, -3, 0, 1, -1, -3, 1, 0, -3, 1, 2, 0, 3, 2, -2, 2, 1, 0, 2, 1, 2, 0]

# Harmonic 322
# payoff = [-1, 2.33333333333333, -8, 2.66666666666667, 0, -2.33333333333333, -4, 2.33333333333333, -3, 1, -3, 1, -11.0000000000000, 0, -2, -2, -1, 3, 0, -2, 1, 1, 1, -2, 7, -1, -2, -2, -2, 1, -2, 2, 2, -3, -3, 3]

u_pure =  reshape(payoff, (A..., N))

# ------------------------------------------------ # ------------------------------------------------ # ------------------------------------------------ 

# for (f, a) in enumerate(pures)
#     for i in players
#         F = f + nPures * (i-1)
#         u_pure[a..., i] = payoff[F]
#     end
# end

for i in players
    payoff_player = payoff[ (i-1) * nPures + 1 : i * nPures ]
    for (f, a) in enumerate(pures)
        u_pure[a... , i] = payoff_player[f]
    end
end


# ------------------------------------------------ # ------------------------------------------------ # ------------------------------------------------ 

# u_pure=[7 1; 2 7;;; -29 24; -6 0;;;; -15 -10; -3 2;;; 23 0; -9 4;;;; -8 1; 4 -6;;; -8 0; -6 5]
# x_start = [ [ 0.3, 0.7 ], [ 0.1, 0.9 ], [ 0.4, 0.6 ] ]

#-----------------------------------
@info "Initial strategy $x_start"

"The operator ... in Julia is like * in Python; when used in function call, unpacks argument.
Payoff tensor is accessed as u[a1, a2, a3, i] where (a1, a2, a3) is pure strategy profile and i is player.
So access payoff tensor as u_pure[a..., i] where a in pures is pure strategy profile, and i in players is player"



# Show payoffs
j = 1
for i in players
    for a in pures
        # println("$j: Player $i, pure $([ai-1 for ai in a]), utility =  $(u_pure[a..., i])")
        println("$j: Player $i, pure $([ai for ai in a]), utility =  $(u_pure[a..., i])")
        global j = j+1
    end
end
#######




# mixed extension of utility function
function u(x, i)
    "x = (x_i) = ( (x_iai) ) is mixed strategy profile in packed format; each xi is mixed strategy of player i" 
    expected_utility_1 = sum( u_pure[a... ,i] * prod( x[j][a[j]] for j in players )  for a in pures )

    # equivalent from the library package, giving array indexed by player
    # expected_utility_2 = StrategicGames.expected_payoff(u_pure, x)[i] # ------------------------------------------------------------------------ leverage more StrategicGame package
end



test_player = 2
@info "Payoff player $test_player:  $( u( x_start , test_player ) ) "




# individual differentials field
function v(x, i, ai)
    "evaluate as v_{i, ai} (x) = u_i (a_i, x_{-i})"

    "Each xi has two entries, adding up to 1.
    For player i, the pure strategy ai can be either 1 or 2.
    For player i, set xi to be the degenerate pure strategy ai, that is 
    if ai = 1 then xi = (1,0)
    elif ai = 2 then xi = (0,1)"

   
    new_xi = zeros( A[i] )
    new_xi[ai] = 1

    # Insert xi at the right place, leaving the mixed strategies of the other players untouched
    new_x = copy(x)
    new_x[i] = new_xi

    # Return expected utility
    u(new_x,i)
end

function payfield(x)
    [ [   v(x, i, ai)   for ai in pures_play[i] ] for i in players ]
end

for i in players
    println("\nPlayer $i")
    println( "u_$i(x) = $(u(x_start, i))" )
    for ai in pures_play[i]
        println( " v_($i, $ai)(x) =  $(v(x_start, i, ai))" )
    end
end
# println("Reduced payoff at x = $(reduced_v(x_start))")
#### End example

# Full replicator dynamics
function RD(x)
    [ [ x[i][ai] * ( v(x, i, ai)  - u(x,i) ) for ai in pures_play[i] ] for i in players ]
end
@info "RD at x_start = $(RD(x_start))"


# ODE replicator

function flatten(x)
    vcat(x...)
end

# Pack; works for any skeleton thanks to cumsum
function pack(flat_x)
    # A and N are global
    num_players = N
    indices = cumsum([0; A])
    [flat_x[indices[i]+1:indices[i+1]] for i in 1:length(A)]
end

#@info "compare x_start and pack(flatten(x_start))"
#@info x_start
#@info pack(flatten(x_start))
# --------------------------------

# Full replicator dynamics with args to feed to ode solver
function RD_ode(flat_x, p, t)
    x = pack(flat_x)
    packed_update = RD(x)
    flat_update = flatten(packed_update)
end



# Continuous replicator update

function ode_RD_dynamics(x0, tspan)
    x0 = flatten(x0)
    println("Initial point: $x0")
    println("First RD update: $(RD_ode(x0, 0, 0))")
    prob = ODEProblem(RD_ode, x0, tspan)
    soln = solve(prob, Tsit5(), reltol = ODE_TOLERANCE, abstol = ODE_TOLERANCE)
    # traj = hcat(soln.u[:]...)'
    traj = soln.u[:]
    traj_packed = [pack(x) for x in traj]

    # @info "---"
    # println(traj_packed[1])
    # println(traj_packed[2])
    # println(traj_packed[3])

    count_returns(traj, x0, RETURN_TOLERANCE)
    return traj_packed
    # x = traj[:, 1]
    # y = traj[:, 3]
    # z = traj[:, 5]
    # return x, y, z

end

# --------------------------------------------
# entropic prox map
function prox(x, y, step)
    # [[ x[i][ai] * exp( step * v(x, i, ai) ) for ai in pures_play[i] ] / sum([ x[i][ai] * exp( step * v(x, i, ai) ) for ai in pures_play[i] ]) for i in players ]
    [[ x[i][ai] * exp( step * y[i][ai] ) for ai in pures_play[i] ] / sum([ x[i][ai] * exp( step * y[i][ai] ) for ai in pures_play[i] ]) for i in players ]
end

# FTRL vanilla
function ftrl(x0, num_iterations, step)
    x = x0
    trajectory = [x0]

    for t in 1:num_iterations
        x = prox(x, payfield(x), step)
        push!(trajectory, x)
    end

    # returns packed trajectory
    return trajectory
end

# HERE, TO TUNE OPTIMISTIC -----------------------------------------------------------------------------------------------------------------------------------------------
function optimistic_ftrl(x0, num_iterations, initial_step)
    x = x0
    trajectory = [x0]
    step = initial_step

    # Initialize exploration step
    x_lead = prox(x0, payfield(x0), step)

    for t in 1:num_iterations
        v_lead = payfield(x_lead)                # unique vector call of the iteration
        x = prox(x, v_lead, step)           # update main point
        x_lead = prox(x, v_lead, step)    # update exploration point using same vector as main point
        push!(trajectory, x)
        # step = step / t                     # update step size
    end

    return trajectory
end
# --------------------------------------------

function plot_trajectory(traj_packed, players, pures, label; color, new = true, multiple = false)

    i, j, k = players
    ai, bj, ck = pures

    X = [ x[i][ai] for x in traj_packed ]
    Y = [ x[j][bj] for x in traj_packed ] 
    Z = [ x[k][ck] for x in traj_packed ] 
    
    if new
        scatter( [  [X[1]], [Y[1]], [Z[1]]  ]..., label = "Initial point", color = "yellow")
    end

    if multiple
        scatter!( [  [X[1]], [Y[1]], [Z[1]]  ]..., label = "Initial point", color = "yellow")
    end

    plot!(X, Y, Z, label = label,  xlims=(0, 1), ylims=(0,1), zlims = (0,1), linewidth = 0.2, color = color)

    
    scatter!( [  [X[end]], [Y[end]], [Z[end]]  ]..., label = "Final point $label", color = color)
    @info "Final point $label: $(X[end]), $(Y[end]), $(Z[end]) )"
    
end


function count_returns(traj_flat, x0, tolerance)
    returns = 0
    for x in traj_flat
        if euclidean(x0, x) < tolerance
            returns += 1
        end
    end
    @info "Number of returns to initial point within tolerance $tolerance: $returns"
end

function save_experiment(experiment_name, comment)
    mkdir(experiment_name)
    open("$experiment_name/$experiment_name.txt","a") do io
        println(io,"u_pure=",u_pure)
        println(io,"comment=",comment)
    end
    savefig("$experiment_name/$experiment_name.pdf")
end

#---------------------------------------------------------------------------------------------------------

# Experiments

# euler_primal_trajectory = euler_RD_dynamics(x_start, 2000, 0.005)
# mirror_descent_trajectory = mirror_descent(x_start, 10000, 0.0001)


ODE_HORIZON = 50
FTRL_HORIZON = 15000

PLAYERS_TO_PLOT = [1, 2, 3]
PURES_TO_PLOT = [1, 1, 1]


ODE_TOLERANCE = 1e-7
RETURN_TOLERANCE = 0.1

FTRL_STEP = 0.01




#---------------------------------------------------------------------------------------------------------

ode_RD_trajectory = ode_RD_dynamics(x_start, (0, ODE_HORIZON))

# VANILLA_FTRL = ftrl(x_start, FTRL_HORIZON, FTRL_STEP)

OPT_FTRL = [optimistic_ftrl(x0, FTRL_HORIZON, FTRL_STEP) for x0 in INITIAL_POINTS]


#---------------------------------------------------------------------------------------------------------


#plot_trajectory(VANILLA_FTRL, PLAYERS_TO_PLOT, PURES_TO_PLOT, "Vanilla"; new = false, color = "red")

plot_trajectory(OPT_FTRL[1], PLAYERS_TO_PLOT, PURES_TO_PLOT, "Optimistic"; new = true, multiple = false,  color = "blue")
[plot_trajectory(traj, PLAYERS_TO_PLOT, PURES_TO_PLOT, "Optimistic"; new = false, multiple = true, color = "blue") for traj in OPT_FTRL[2:end]]
plot_trajectory(ode_RD_trajectory, PLAYERS_TO_PLOT, PURES_TO_PLOT, "RD"; new = false, multiple = true, color = "black" )

# ! is to re-use the same canva
# plot!(euler_primal_trajectory..., label = "Euler in primal space", title = "(Full) RD on harmonic 2x2x2 game")
# plot!(mirror_descent_trajectory..., label = "Mirror descent",  linestyle=:dash, lw = 2)

# Plot cube
plot!([1,1],[1,1], [0,1], color = "black", label = false, legend = false) 
plot!([0,0],[1,1], [0,1], color = "black", label = false) 
plot!([0,1],[1,1], [1,1], color = "black", label = false) 
plot!([0,1],[0,0], [1,1], color = "black", label = false)  
plot!([0,1],[1,1], [0,0], color = "black", label = false) 
plot!([1,1],[1,0], [1,1], color = "black", label = false)  
plot!([0,0],[1,0], [1,1], color = "black", label = false)  
plot!([0,0],[1,0], [0,0], color = "black", label = false) 
plot!([1,1],[0,0], [0,1], color = "black", label = false)


# Plot initial condition
# a = x_start[1][1]
# b = x_start[2][1]
# c = x_start[3][1]
# scatter!( [  [a], [b], [c]  ]..., label = "Initial point")

save_experiment("generalizedHarmonic_223_multiple_ODE$(join(PURES_TO_PLOT, "-"))", "Generalized harmonic, measure [  [1, 2], [1, 3], [1,3,4]  ]")



