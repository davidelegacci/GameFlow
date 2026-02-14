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

println("\n START")

# to save figure in current folder
cd(dirname(@__FILE__()))

# using StrategicGames
using DifferentialEquations
using Plots


"""
Implementattion of 3 players  [2,2,2] finite game in normal form and replicator dynamics
"""

#  Number of players
N = 3
players = 1:N

# Number of actions for each player
A = [2,2,2]

# Pure strategies
pures_play = [ 1:A[i] for i in players ]

pures = [(a1, a2, a3) for a1 in pures_play[1], a2 in pures_play[2], a3 in pures_play[3]]

# random payoff tensor as 2 x 2 x 2 x 3 tensor
# u_pure = rand(-5:5, A..., N)

# Reshape and permutation for compatibility with ordering of payoff output by Candogan code, generate_harmonic.py, mixed_extension.py, etc. in 2x2x2 case.
# Swap first and third strategy index, invariant second strategy index and player index, achieved by permutedims(... , (3, 2, 1, 4))
# payoff = [-4, -3, 33, -29, 1, -4, 9, -9, 1, -8, -10, -1, 0, 7, 9, 2, -12, -6, -10, 3, 3, -1, 5, -10]
# payoff = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# payoff = [-9, 1, 10, 1, 3, -1, 8, -7, -2, 8, 5, -9, -5, -2, 5, -2, 27, 8, -10, -1, -6, -4, -5, 3]
# u_pure = permutedims( reshape(payoff, (A..., N)), (3, 2, 1, 4) )
u_pure=[7 1; 2 7;;; -29 24; -6 0;;;; -15 -10; -3 2;;; 23 0; -9 4;;;; -8 1; 4 -6;;; -8 0; -6 5]

"The operator ... in Julia is like * in Python; when used in function call, unpacks argument.
Payoff tensor is accessed as u[a1, a2, a3, i] where (a1, a2, a3) is pure strategy profile and i is player.
So access payoff tensor as u_pure[a..., i] where a in pures is pure strategy profile, and i in players is player"

# Show payoffs
j = 1
for i in players
    for a in pures
        println("$j: Player $i, pure $([ai-1 for ai in a]), utility =  $(u_pure[a..., i])")
        global j = j+1
    end
end
#######

# mixed extension of utility function
function u(x, i)
    "x = (x1, x2, x3) = ( (x1,1, x1,2), (x2,1, x2,2), (x3,1, x3,2) ) is mixed strategy profile; each xi is mixed strategy of player i"
    expected_utility = sum( u_pure[a... ,i] * x[1][a[1]] * x[2][a[2]] * x[3][a[3]] for a in pures )

    # equivalent from the library package, giving array indexed by player
    # expected_utility = StrategicGames.expected_payoff(u_pure, x)
end

# individual differentials field
function v(x, i, ai)
    "evaluate as v_{i, ai} (x) = u_i (a_i, x_{-i})"

    "Each xi has two entries, adding up to 1.
    For player i, the pure strategy ai can be either 1 or 2.
    For player i, set xi to be the degenerate pure strategy ai, that is 
    if ai = 1 then xi = (1,0)
    elif ai = 2 then xi = (0,1)"

    xi = [0,0]
    xi[ai] = 1

    # Insert xi at the right place, leaving the mixed strategies of the other players untouched
    if i == 1
        x = [ xi, x[2], x[3] ]
    elseif i == 2
        x = [ x[1], xi, x[3] ]
    elseif i == 3
        x = [ x[1], x[2], xi ]
    end

    # Return expected utility
    u(x,i)
end

# function reduced_v(x)
#     [ v(x, i, 2) - v(x, i, 1) for i in players ]
# end

#### Example: build mixed strategy profile
a = 0.2
b = 0.3
c = 0.4

x1_test = [a, 1-a ]  # One particular mixed-strategy emploied by player 1
x2_test = [b, 1-b]   # One particular pure-strategy emploied by player 2
x3_test = [c, 1-c]   # One particular mixed-strategy emploied by partner 3



x_test = [x1_test,x2_test,x3_test]     # A strategy profile



println("Mixed strategy x = $x_test")
for i in players
    println("\nPlayer $i")
    println( "u_$i(x) = $(u(x_test, i))" )
    for ai in pures_play[i]
        println( " v_($i, $ai)(x) =  $(v(x_test, i, ai))" )
    end
end
# println("Reduced payoff at x = $(reduced_v(x_test))")
#### End example


# Full replicator dynamics
function RD(x)
    [ [ x[i][ai] * ( v(x, i, ai)  - u(x,i) ) for ai in pures_play[i] ] for i in players ]
end
println("RD at x = $(RD(x_test))")

function payfield(x)
    [ [   v(x, i, ai)   for ai in pures_play[i] ] for i in players ]
end

function prox(x, step)
    [[ x[i][ai] * exp( step * v(x, i, ai) ) for ai in pures_play[i] ] / sum([ x[i][ai] * exp( step * v(x, i, ai) ) for ai in pures_play[i] ]) for i in players ]
end


# Euler replicator update
function euler_RD_dynamics(x0, num_iterations, step)
    x = x0
    trajectory = [x0]

    for t in 1:num_iterations
        x = x + step * RD(x)
        push!(trajectory, x)
    end

    # println( [round.(xi, digits = 3) for xi in x] )
    x = [p[1][1] for p in trajectory]
    y = [p[2][1] for p in trajectory]
    z = [p[3][1] for p in trajectory]
    return x, y, z
end

# EW  update
function mirror_descent(x0, num_iterations, step)
    x = x0
    trajectory = [x0]

    for t in 1:num_iterations
        x = prox(x, step)
        push!(trajectory, x)
    end

    # println( [round.(xi, digits = 3) for xi in x] )
    x = [p[1][1] for p in trajectory]
    y = [p[2][1] for p in trajectory]
    z = [p[3][1] for p in trajectory]
    return x, y, z
end

# ODE replicator

function flatten(x)
    vcat(x...)
end

# Pack; works only for all players with same number of strategies
function pack(flat_x)
    # A and N are global
    Ai = A[1] # = 2
    num_players = N # = 3
    [flat_x[(i-1) * Ai+1 : i * Ai] for i in 1:num_players]
end

# Full replicator dynamics with args to feed to ode solver
function RD_ode(flat_x, p, t)
    x = pack(flat_x)
    packed_update = RD(x)
    flat_update = flatten(packed_update)
end

# Just payfield (not projected)
function payfield_ode(flat_x, p, t)
    x = pack(flat_x)
    packed_update = payfield(x)
    flat_update = flatten(packed_update)
end

# Continuous replicator update

function ode_RD_dynamics(x0, tspan)
    x0 = flatten(x0)
    println("Initial point: $x0")
    println("First payfield: $(payfield_ode(x0, 0, 0))")
    println("First RD update: $(RD_ode(x0, 0, 0))")
    prob = ODEProblem(RD_ode, x0, tspan)
    soln = solve(prob, Tsit5(), reltol = 1e-5, abstol = 1e-5)
    traj = hcat(soln.u[:]...)'
    x = traj[:, 1]
    y = traj[:, 3]
    z = traj[:, 5]
    return x, y, z

end

function save_experiment(experiment_name, comment)
    mkdir(experiment_name)
    open("$experiment_name/$experiment_name.txt","a") do io
        println(io,"u_pure=",u_pure)
        println(io,"comment=",comment)
    end
    savefig("$experiment_name/$experiment_name.pdf")
end

############################
############################

# Experiments



euler_primal_trajectory = euler_RD_dynamics(x_test, 2000, 0.005)
mirror_descent_trajectory = mirror_descent(x_test, 10000, 0.0001)

ode_trajectory = ode_RD_dynamics(x_test, (0, 10))

plot(ode_trajectory..., label = "ODE",  xlims=(0, 1), ylims=(0,1), zlims = (0,1))

# ! is to re-use the same canva
plot!(euler_primal_trajectory..., label = "Euler in primal space", title = "(Full) RD on harmonic 2x2x2 game")
plot!(mirror_descent_trajectory..., label = "Mirror descent",  linestyle=:dash, lw = 2)

# Plot cube
plot!([1,1],[1,1], [0,1], color = "black", label = false) 
plot!([0,0],[1,1], [0,1], color = "black", label = false) 
plot!([0,1],[1,1], [1,1], color = "black", label = false) 
plot!([0,1],[0,0], [1,1], color = "black", label = false)  
plot!([0,1],[1,1], [0,0], color = "black", label = false) 
plot!([1,1],[1,0], [1,1], color = "black", label = false)  
plot!([0,0],[1,0], [1,1], color = "black", label = false)  
plot!([0,0],[1,0], [0,0], color = "black", label = false) 
plot!([1,1],[0,0], [0,1], color = "black", label = false)

# Plot initial condition
scatter!( [  [a], [b], [c]  ]..., label = "Initial point")

# save_experiment("prox_harmonic_full", "In full implementation, prox behaves like ODE on harmonic game; naive Euler spirals out")



