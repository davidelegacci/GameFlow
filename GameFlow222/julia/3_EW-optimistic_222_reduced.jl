# to save figure in current folder
cd(dirname(@__FILE__()))

# using StrategicGames
# using DifferentialEquations
using Plots

"Exponential Weights on finite [2, 2, 2] normal form game. Work in reduced coordinates and discrete setting.

At the moment only entropic prox map implemented, but very easy to adapt to generic FTRL just changing prox map, everything else is the same

Computes 3 trajectories in discrete time, with entropic regularizer:

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

########################################################
# Initial point
########################################################

# initial_point =  [0.7823802453721485, 0.8175106033034041, 0.9738153265749638]
# initial_point =  [0.6582068432777008, 0.024494805388966312, 0.9940742768758856]

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
payoff = [7, -29, 1, 24, 2, -6, 7, 0, -15, 23, -10, 0, -3, -9, 2, 4, -8, -8, 1, 0, 4, -6, -6, 5]
# payoff = [-9, 1, 10, 1, 3, -1, 8, -7, -2, 8, 5, -9, -5, -2, 5, -2, 27, 8, -10, -1, -6, -4, -5, 3]
# payoff = [11, -7, -33, 23, -6, 5, -3, -2, -48, 38, -5, 0, 9, -8, -7, 3, 34, 8, -7, 6, 6, 5, -7, 7]

########################################################
# Potential in Candogan flat format
########################################################
# payoff = [-14, 13, -18, -8, -8, 8, -7, 1, -16, 6, 2, -1, -16, 0, 7, 7, -7, 8, 2, -8, 0, 4, 8, -4]
# payoff = [5, 28, 5, 5, -6, 5, -3, -2, 2, 5, -5, 0, -3, -8, -7, 3, -3, 8, -7, 6, 6, 5, -7, 7]

########################################################
# Potential-Harmonic mixture in Candogan flat format
########################################################
payoff_h = [-21, -16, 2, 21, -4, -8, -9, 7, 6, 24, 9, -4, -7, -8, 3, 7, 20, 0, -7, 7, 2, 9, -2, -3]
payoff_p = [-7, -10, -39, -8, -4, -8, -9, 7, 13, -6, 9, -4, -20, -8, 3, 7, -8, 0, -7, 7, 2, 9, -2, -3]

par = 0

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
# KEY: Reduced prox mapping with entropic distance generating function
########################################################
"Obtained pulling back the function x * exp(y) / sum [ x * exp(y) ] to simplex in usual way"
function prox(x, y,  step)
    num = x .* exp.( step * y )
    den = ones(3) + num - x
    num ./ den
end

########################################################
# Algorithms
########################################################
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
function save_experiment(experiment_name, comment)
    "Create folder with experiment name, save image and text file with comment"
    dir = "experiments/NeurIPS2024/$experiment_name"
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
end

function plpoint(point, label, color)
    "Plot single point"
    scatter!( [  [a] for a in point  ]..., label = label, color = color, markersize = 8)
end

function plot_algo(algo, algo_name; x0, num_iterations = 1e3, initial_step = 0.01, color, ls = :solid)
    "Compute and plot trajectory from algorithm"
    traj = algo( x0, num_iterations, initial_step )
    final_point = [ traj[1][end], traj[2][end] , traj[3][end] ]
    @info "Final $algo_name: $final_point" 
    plot!(traj..., label = algo_name, color = color, ls = ls)
    plpoint(final_point, "$algo_name final point", color)
end

########################################################
# Experiments
########################################################

@info "Payoff"
see_payoffs()
@info "Initial: $initial_point" 

# Init figure
plot(xlims=(0, 1), ylims=(0,1), zlims = (0,1), title = "Extra gradient", size=(800,800), margin = -100Plots.mm)

# Plot initial point and barycenter
plpoint(initial_point, "Initial point",  "yellow")
NE = [0.5, 0.5, 0.5]
plpoint(NE, "Center",  "black")


# Plot trajectories
plot_algo(mirror_descent, "Learning", x0 = initial_point, color = "blue", num_iterations =  2e3)
plot_algo(extra_gradient_mirror_descent, "Extra gradient", x0 = initial_point, num_iterations = 5e4, color =  "green")
plot_algo(optimistic_mirror_descent, "Optimistic learning", x0 = initial_point, num_iterations =  8e4, color = "red", ls = :solid)


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

# save_experiment("extra_gradient_EW_harmonic", "Cool extra gradient mirror descent with entropic regularixer convergence in harmonic 2x2x2")




