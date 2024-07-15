# to save figure in current folder
cd(dirname(@__FILE__()))

CURRENT_FOLDER = "exploring_harmonic_dynamics"

# using StrategicGames
using DifferentialEquations
using Plots
using LinearAlgebra
using StatsBase

"
- FTRL on finite [2, 2, 2] normal form game
- Work in reduced coordinates
- entropic regularizer

0. continuous time (replicator dynamics)
1. vanilla mirror descent
2. extra gradient mirror descent
3. optimistic mirror descent

In more detail:
--------------------------------
1. vanilla mirror descent with entropic distance generating function, i.e. exponential weights

y --> y + step * v(x)
x = Q(y)
Q choice or mirror map

equivalent to

x --> P( x, y = step * v(x) )
P prox map
--------------------------------
2. Extra gradient: Like mirror descent, but with 2 'dual vector' queries per iteration
Given x0

x_exp = P( x0, v(x0) )
x1 = P(x0, v(x_exp))
--------------------------------
3: Optimistic: like extra gradient, but use previous dual vector rather than calling two times per iteration

Given x0

x_exp = P( x0, v_previous )
x1 = P(x0, v(x_exp))
--------------------------------
"

println("start")

########################################################
# Skeleton; must be 2x2x2
########################################################
#  Number of players
N = 3
# Number of actions for each player
A = [2,2,2]

BARYCENTER = [0.5, 0.5, 0.5]

########################################################
# Initial point
########################################################

# initial_point =  [0.7823802453721485, 0.8175106033034041, 0.9738153265749638]
# initial_point =  [0.6582068432777008, 0.024494805388966312, 0.9940742768758856]
# initial_point = [0.5668528500530363, 0.09562312642919535, 0.6790207264606783]

### Random initial point
initial_point = rand(3)

########################################################
# Payoff
########################################################

" Payoff in current format is is 2 x 2 x 2 x 3 tensor. Need reshape and permutation for compatibility with ordering of payoff output by Candogan code, generate_harmonic.py, mixed_extension.py, etc. in 2x2x2 case.
- Swap first and third strategy index, invariant second strategy index and player index, achieved by permutedims(... , (3, 2, 1, 4))
- reshape between flat and tensor
"
# To convert payoff output by candogan cone in current format: pack in tensor, and permute
function candogan_to_julia(payoff)
    permutedims( reshape(payoff, (A..., N)), (3, 2, 1, 4) )
end

# To convert payoff in current format to feed in Candoan code: permute, and flatten
# function julia_to_candogan(payoff)
#     reshape(permutedims( payoff, (3, 2, 1, 4) ), 24)
# end

########################################################
# Harmonic in Candogan flat format
########################################################
# payoff = [7, -29, 1, 24, 2, -6, 7, 0, -15, 23, -10, 0, -3, -9, 2, 4, -8, -8, 1, 0, 4, -6, -6, 5]
# payoff = [-9, 1, 10, 1, 3, -1, 8, -7, -2, 8, 5, -9, -5, -2, 5, -2, 27, 8, -10, -1, -6, -4, -5, 3]
# payoff = [11, -7, -33, 23, -6, 5, -3, -2, -48, 38, -5, 0, 9, -8, -7, 3, 34, 8, -7, 6, 6, 5, -7, 7]
# payoff = [6, 0, -5, -1, 1, 3, -3, -1, 3, -8, -1, -2, 0, 1, -2, 1, -10, -1, 3, -3, 1, -2, 0, 0]

# payoff =  [-5.00000000000000, 3.00000000000000, 1.00000000000000, -1.00000000000000, -3, -2, 2, 1, 3.00000000000000, -7.00000000000000, 0, -1, -2, 3, -3, 1, -3.81000000000000, -3, 2, -2, -1, 2, 3, 3]
payoff = [16, 16, 24, -9, -1, 3, 3, 2, 17, -18, 3, -3, -1, 1, -4, -4, -23, -4, -4, 3, 2, 0, 3, -3]
EXPERIMENT_NAME = "generalized_harmonic_222"

########################################################
# Potential in Candogan flat format
########################################################
# payoff = [-14, 13, -18, -8, -8, 8, -7, 1, -16, 6, 2, -1, -16, 0, 7, 7, -7, 8, 2, -8, 0, 4, 8, -4]
# payoff = [5, 28, 5, 5, -6, 5, -3, -2, 2, 5, -5, 0, -3, -8, -7, 3, -3, 8, -7, 6, 6, 5, -7, 7]

########################################################
# Potential-Harmonic mixture in Candogan flat format
########################################################
# payoff_h = [-21, -16, 2, 21, -4, -8, -9, 7, 6, 24, 9, -4, -7, -8, 3, 7, 20, 0, -7, 7, 2, 9, -2, -3]
# payoff_p = [-7, -10, -39, -8, -4, -8, -9, 7, 13, -6, 9, -4, -20, -8, 3, 7, -8, 0, -7, 7, 2, 9, -2, -3]

# par = 0

# payoff = par * payoff_p + (1-par) * payoff_h

# Convert to tensor
u_pure = candogan_to_julia(payoff)

########################################################
# Random payoff tensor as 2 x 2 x 2 x 3 tensor
########################################################
# u_pure = rand(-5:5, A..., N)

########################################################
# Input in Julia tensor format
########################################################
# u_pure = [1 -3; 2 -4;;; -3 -5; 2 2;;;; 3 0; 2 5;;; -4 -2; -4 1;;;; -2 4; -5 1;;; -3 5; 3 -3]

# For Tolosa abstract
# u_pure=[7 1; 2 7;;; -29 24; -6 0;;;; -15 -10; -3 2;;; 23 0; -9 4;;;; -8 1; 4 -6;;; -8 0; -6 5]

####################################################################
# Begin algorithm
####################################################################
"
NOTATION COMPATIBILITY

Now payoff is accessed indexing u_pure[a1, a2, a3, i], with strategy indices starting at 1 (since we're using Julia), and player index at the end.

The output of mixed_extension.py and notes on paper work with strategy indices starting at zero, and player index at the beginning.

To make it easier to write down coefficients, write little helper function converting between the two notations. 

Syntax
u_pure[a1, a2, a3, i] == u_func(i, (a1-1, a2-1, a3-1)) for ai in [1, 2]
u_pure[a1+1, a2+1, a3+1, i] == u_func(i, (a1, a2, a3)) for ai in [0, 1]
"
function u_func(i, a)
    a1, a2, a3 = a
    u_pure[a1+1, a2+1, a3+1, i]
end

# Sanity check: print explicitely payoff functions
function see_payoffs()
    players = 1:N
    # Pure strategies
    pures_play = [ 1:A[i] for i in players ]
    pures = [(a1, a2, a3) for a1 in pures_play[1], a2 in pures_play[2], a3 in pures_play[3]]
    j = 1
    for i in players
        for a in pures
            println("$j: Player $i, pure $([ai-1 for ai in a]), utility =  $(u_pure[a..., i])")
            j = j+1

            # To check that conversion function is fine
            # a1, a2, a3 = a
            # @assert u_pure[a1, a2, a3, i] == u_func(i, (a1-1, a2-1, a3-1))
        end
    end
end

function write_payoffs(io)
    "Use to write payoffs in txt when saving experiment"
    players = 1:N
    # Pure strategies
    pures_play = [ 1:A[i] for i in players ]
    pures = [(a1, a2, a3) for a1 in pures_play[1], a2 in pures_play[2], a3 in pures_play[3]]
    j = 1
    for i in players
        for a in pures
            println(io, "$j: Player $i, pure $([ai-1 for ai in a]), utility =  $(u_pure[a..., i])")
            j = j+1

            # To check that conversion function is fine
            # a1, a2, a3 = a
            # @assert u_pure[a1, a2, a3, i] == u_func(i, (a1-1, a2-1, a3-1))
        end
    end
end

########################################################
# Reduced payoff field
########################################################

a1 = u_func(1, (0,0,0))
b1 = u_func(1, (0,0,1))
c1 = u_func(1, (0,1,0))
d1 = u_func(1, (0,1,1))
e1 = u_func(1, (1,0,0))
f1 = u_func(1, (1,0,1))
g1 = u_func(1, (1,1,0))
h1 = u_func(1, (1,1,1))

a2 = u_func(2, (0,0,0))
b2 = u_func(2, (0,0,1))
c2 = u_func(2, (0,1,0))
d2 = u_func(2, (0,1,1))
e2 = u_func(2, (1,0,0))
f2 = u_func(2, (1,0,1))
g2 = u_func(2, (1,1,0))
h2 = u_func(2, (1,1,1))

a3 = u_func(3, (0,0,0))
b3 = u_func(3, (0,0,1))
c3 = u_func(3, (0,1,0))
d3 = u_func(3, (0,1,1))
e3 = u_func(3, (1,0,0))
f3 = u_func(3, (1,0,1))
g3 = u_func(3, (1,1,0))
h3 = u_func(3, (1,1,1))

function v(x)
    "Reduced payoff field"
    x1, x2, x3 = x

    V11 = a1 * x2 * x3 + b1 * x2 * (1-x3) + c1 * (1-x2) * x3 + d1 * (1-x2) * (1-x3)
    V12 = e1 * x2 * x3 + f1 * x2 * (1-x3) + g1 * (1-x2) * x3 + h1 * (1-x2) * (1-x3)
    v1 = V12 - V11

    V21 = a2 * x1 * x3 + b2 * x1 * (1-x3) + e2 * (1-x1) * x3 + f2 * (1-x1) * (1-x3)
    V22 = c2 * x1 * x3 + d2 * x1 * (1-x3) + g2 * (1-x1) * x3 + h2 * (1-x1) * (1-x3)
    v2 = V22 - V21

    V31 = a3 * x1 * x2 + c3 * x1 * (1-x2) + e3 * (1-x1) * x2 + g3 * (1-x1) * (1-x2)
    V32 = b3 * x1 * x2 + d3 * x1 * (1-x2) + f3 * (1-x1) * x2 + h3 * (1-x1) * (1-x2)
    v3 = V32 - V31

    [v1, v2, v3]
end

########################################################
# Dynamics
########################################################

#-------------------------------------------------------
## Replicator dynamics
#-------------------------------------------------------
function RD(x, p, t)
    # p and t are dummy for ODE solver
    # if needed put here global minus sign
    x .* ( ones(3) .- x) .* v(x)
end


#-------------------------------------------------------
## Reduced prox mapping with entropic distance generating function
#-------------------------------------------------------
"Obtained pulling back the function x * exp(y) / sum [ x * exp(y) ] to simplex in usual way"
function prox(x, y,  step)
    num = x .* exp.( step * y )
    den = ones(3) + num - x
    num ./ den
end


#-------------------------------------------------------
## Vanilla Mirror Descent
#-------------------------------------------------------
function mirror_descent(x0, num_iterations, initial_step)
    x = x0
    trajectory = [x0]
    step = initial_step

    for t in 1:num_iterations
        x = prox(x, v(x), step)
        push!(trajectory, x)
        # step = step / t           # update step size
    end

    # Convert between list of points and list of coordinates
    x1 = [x[1] for x in trajectory]
    x2 = [x[2] for x in trajectory]
    x3 = [x[3] for x in trajectory]
    return x1, x2, x3
end

#-------------------------------------------------------
## Extra Gradient Mirror Descent
#-------------------------------------------------------
function extra_gradient_mirror_descent(x0, num_iterations, initial_step)
    x = x0
    trajectory = [x0]
    step = initial_step

    for t in 1:num_iterations
        x_explor = prox(x, v(x), step)     # exploration step, first vector call
        x = prox(x, v(x_explor), step)     # main step, second vector call
        push!(trajectory, x)
        # step = step / t                  # update step size
    end

    x1 = [x[1] for x in trajectory]
    x2 = [x[2] for x in trajectory]
    x3 = [x[3] for x in trajectory]
    return x1, x2, x3
end

#-------------------------------------------------------
## Optimistic Mirror Descent
#-------------------------------------------------------
function optimistic_mirror_descent(x0, num_iterations, initial_step)
    x = x0
    trajectory = [x0]
    step = initial_step

    # Initialize exploration step
    x_explor = prox(x0, v(x0), step)

    for t in 1:num_iterations
        v_explor = v(x_explor)                # unique vector call of the iteration
        x = prox(x, v_explor, step)           # update main point
        x_explor = prox(x, v_explor, step)    # update exploration point using same vector as main point
        push!(trajectory, x)
        # step = step / t                     # update step size
    end

    x1 = [x[1] for x in trajectory]
    x2 = [x[2] for x in trajectory]
    x3 = [x[3] for x in trajectory]
    return x1, x2, x3
end


########################################################
# Utils
########################################################
function ode_RD_dynamics(x0, tspan)
    prob = ODEProblem(RD, x0, tspan)
    soln = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
    traj = hcat(soln.u[:]...)'
    x1 = traj[:, 1]
    x2 = traj[:, 2]
    x3 = traj[:, 3]
    return x1, x2, x3

end

function save_experiment(experiment_name, comment)
    "Create folder with experiment name, save image and text file with comment"
    dir = "experiments/$CURRENT_FOLDER/$experiment_name"
    mkdir(dir)
    open("$dir/$experiment_name.txt","a") do io
        println(io,"Payoff Julia format = ", u_pure)
        println(io,"\nPayoff Candogan format = ", payoff)
        println(io, "")
        write_payoffs(io)
        println(io, "\nInitial point = ", initial_point )
        println(io,"\ncomment = ", comment)
    end
    savefig("$dir/$experiment_name.pdf")
    savefig("$dir/$experiment_name.png")
end

function save_distance(experiment_name, comment)
    "Create folder with experiment name, save image and text file with comment"
    dir = "experiments/$CURRENT_FOLDER/$experiment_name"
    savefig("$dir/$experiment_name-distance-$comment.pdf")
    # savefig("$dir/$experiment_name-distance-$comment.png")
end

function plpoint(point, label, color, ms = 5)
    "Plot single point"
    scatter!( [  [a] for a in point  ]..., label = label, color = color, markersize = ms)
end

# function measure_euclidean_distance_from_NE(coords, target)
#     coords_diff = [ target[alpha] .- coords[alpha] for alpha in 1:3 ]
#     time = 1:length( coords_diff[1] )
#     points_diff = [ [ coords_diff[1][t], coords_diff[2][t], coords_diff[3][t] ] for t in time ]
#     distance = [norm(p) for p in points_diff]
#     return distance
# end

function measure_KL_distance_from_NE(coords, target)
    "cumbersome packing unpacking, to improve"
    time = 1:length( coords[1] )
    players = 3

    full_target = [ ( 1-target[i], target[i] ) for i in 1:players ] # list with 3 points, each 2 components adding to 1
    full_points = [  [ [ 1-coords[i][t], coords[i][t] ]   for t in time ]  for i in 1:players  ] # list with 3 lists of points, each 2 components adding up to 1

    KL_distance = [  ]
    for i in 1:players
        NEi = collect(full_target[i]) # convert from tuple to vector
        strats_i = full_points[i]
        di = [ kldivergence(NEi, strat_i) for strat_i in strats_i ]
        push!(KL_distance, di)
    end
    return sum(KL_distance) # sum over players
end



#-------------------------------------------
# Main plotting functions
#-------------------------------------------

# super unefficient, compute target and the re-compute orbit, don't have time to make smarter now
function get_taregt(algo; x0, num_iterations, initial_step = 0.01)
    traj = algo( x0, num_iterations, initial_step )
    target = [ traj[1][end], traj[2][end] , traj[3][end] ]
    @info "TARGET: $target" 
    return target
end


function plot_algo(algo, algo_name; x0, num_iterations = 10, initial_step = 0.1, color, ls = :solid)
    "Compute and plot trajectory from algorithm"
    traj = algo( x0, num_iterations, initial_step ) # list of coordinates
    final_point = [ traj[1][end], traj[2][end] , traj[3][end] ]
    @info "Final $algo_name: $final_point" 
    plot!(traj..., label = algo_name, color = color, ls = ls)
    # plpoint(final_point, "$algo_name final point", color)
    plpoint(final_point, "", color)
    return traj
end

function plot_RD(x0, final_time )
    ode_traj = ode_RD_dynamics(x0, (0, final_time))
    ode1, ode2, ode3 = ode_traj
    plot!(ode_traj..., label = "Continuous time MD", linestyle=:solid, lw = 2, color = "black")

end

########################################################
# Experiments
########################################################




@info "Payoff"
see_payoffs()
@info "Initial: $initial_point" 

# Init figure
plot(xlims=(0, 1), ylims=(0,1), zlims = (0,1), title = "Exponential weights update", size=(800,800), margin = -100Plots.mm)

# Plot initial point and barycenter
plpoint(initial_point, "Initial point",  "yellow")
NE = [0.5, 0.5, 0.5]
# plpoint(NE, "Center",  "black", 2)


#---------------------------------------------------------
# Plot trajectories
#---------------------------------------------------------
NUM_ITER = 5000

# Use as target point reached after 1.5 iterations (re-implement to harvest trajectory from here, not computing twice)
# TARGET = get_taregt(optimistic_mirror_descent, x0 = initial_point, num_iterations = 1.5 * NUM_ITER)
TARGET = BARYCENTER

# plpoint(TARGET, "NE",  "black", 2)
#---------------------------------------------------------
MD = plot_algo(mirror_descent, "Vanilla mirror descent", x0 = initial_point, color = "blue", num_iterations = NUM_ITER)
EGMD = plot_algo(extra_gradient_mirror_descent, "Mirror-prox", x0 = initial_point, num_iterations = NUM_ITER, color =  "green")
OMD = plot_algo(optimistic_mirror_descent, "Optimistic mirror descent", x0 = initial_point, num_iterations = 3 * NUM_ITER, color = "red", ls = :dash)
#------------------------------------------------------------------------------------------------------------------
# ODE solution
#------------------------------------------------------------------------------------------------------------------
plot_RD(initial_point, 1000)
#------------------------------------------------------------------------------------------------------------------


# Plot cube
plot!([1,1], [1,1], [0,1], color = "black", label = false) 
plot!([0,0], [1,1], [0,1], color = "black", label = false) 
plot!([0,1], [1,1], [1,1], color = "black", label = false) 
plot!([0,1], [0,0], [1,1], color = "black", label = false)  
plot!([0,1], [1,1], [0,0], color = "black", label = false) 
plot!([1,1], [1,0], [1,1], color = "black", label = false)  
plot!([0,0], [1,0], [1,1], color = "black", label = false)  
plot!([0,0], [1,0], [0,0], color = "black", label = false) 
plot!([1,1], [0,0], [0,1], color = "black", label = false)

save_experiment(EXPERIMENT_NAME, "Exploring harmonic dynamics")  # <------------------------------------ save


#---------------------------------------------------------
# Measure distance from equilibrium
#---------------------------------------------------------
# distance_MD = measure_euclidean_distance_from_NE( MD, TARGET)
# distance_EGMD = measure_euclidean_distance_from_NE(EGMD, TARGET)
# distance_OMD = measure_euclidean_distance_from_NE(OMD, TARGET)

distance_MD = measure_KL_distance_from_NE( MD, TARGET)
distance_EGMD = measure_KL_distance_from_NE(EGMD, TARGET)
distance_OMD = measure_KL_distance_from_NE(OMD, TARGET)

#---------------------------------------------------------
# Plot distance from equilibrium, with vanilla, linear axes
#---------------------------------------------------------
plot(
    title = "Convergence rate to $TARGET",
    xlabel = "iterations", ylabel = "KL distance from NE",
    xguidefontsize = 6, yguidefontsize = 6, titlefontsize = 8,
    # xscale = :log,
    # yscale = :log,
    )

plot!(distance_EGMD, label = "mirror-prox MD")
plot!(distance_OMD, label = "optimistic MD", ls = :dash)
plot!(distance_MD, label = "vanilla MD")

save_distance(EXPERIMENT_NAME, "vanilla")                     # <------------------------------------ save
#---------------------------------------------------------

# Plot distance from equilibrium, w/o vanilla, linear axes
#---------------------------------------------------------
plot(
    title = "Convergence rate to $TARGET",
    xlabel = "iterations", ylabel = "KL distance from NE",
    xguidefontsize = 6, yguidefontsize = 6, titlefontsize = 8,
    # xscale = :log,
    # yscale = :log,
    )

plot!(distance_EGMD, label = "mirror-prox MD")
plot!(distance_OMD, label = "optimistic MD", ls = :dash)

save_distance(EXPERIMENT_NAME, "no-vanilla_linear")                  # <------------------------------------ save


# Plot distance from equilibrium, w/o vanilla, log Y axe
#---------------------------------------------------------
plot(
    title = "Convergence rate to $TARGET",
    xlabel = "iterations", ylabel = "KL distance from NE",
    xguidefontsize = 6, yguidefontsize = 6, titlefontsize = 8,
    # xscale = :log,
    yscale = :log,
    )

plot!(distance_EGMD, label = "mirror-prox MD")
plot!(distance_OMD, label = "optimistic MD", ls = :dash)

save_distance(EXPERIMENT_NAME, "no-vanilla_log")                  # <------------------------------------ save



println("end")