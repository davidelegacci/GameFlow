"CAREFUL WITH GLOBAL MINUS SIGN"

"Work in reduced coordinates, computing by hand expression for reduced payoff field and reduced prox map."

"Plots 3 trajectories:
- continuous time FTRL with entropic regularizer = RD
- Mirror descent with entropic rgularizer = EW

[DEPRECATED] - Euler discretization in primal space x --> x + step * RD(x) [DISCARDED]

y --> y + step * v(x)
x = Q(y)
Q choice or mirror map

equivalent to

x --> P(x,y)
P prox map
"

# To do: change func structure to feed payoff

println("start")

# to save figure in current folder
cd(dirname(@__FILE__()))

# using StrategicGames
using DifferentialEquations
using Plots

# using LatexStrings


"""
Implementattion of 3 players  [2,2,2] finite game in normal form and replicator dynamics
"""


#  Number of players
N = 3
# Number of actions for each player
A = [2,2,2]


# To check and use these functions
function candogan_to_julia(payoff)
    permutedims( reshape(payoff, (A..., N)), (3, 2, 1, 4) )
end

function julia_to_candogan(payoff)
    reshape(permutedims( payoff, (3, 2, 1, 4) ), 24)
end


# Reshape and permutation for compatibility with ordering of payoff output by Candogan code, generate_harmonic.py, mixed_extension.py, etc. in 2x2x2 case.
# Swap first and third strategy index, invariant second strategy index and player index, achieved by permutedims(... , (3, 2, 1, 4))

# Harmonic
payoff_harmonic = [7, -29, 1, 24, 2, -6, 7, 0, -15, 23, -10, 0, -3, -9, 2, 4, -8, -8, 1, 0, 4, -6, -6, 5]
# u_pure = permutedims( reshape(payoff, (A..., N)), (3, 2, 1, 4) )

# Potential
payoff_potential = [-14, 13, -18, -8, -8, 8, -7, 1, -16, 6, 2, -1, -16, 0, 7, 7, -7, 8, 2, -8, 0, 4, 8, -4]
# u_pure = permutedims( reshape(payoff, (A..., N)), (3, 2, 1, 4) )

# Potential - Harmonic
pot_param = 0.1
payoff = (1 - pot_param) * payoff_harmonic + pot_param * payoff_potential
u_pure = permutedims( reshape(payoff, (A..., N)), (3, 2, 1, 4) )

# Random payoff tensor as 2 x 2 x 2 x 3 tensor
# u_pure = rand(-5:5, A..., N)


# Input in tensor format
# u_pure = [1 -3; 2 -4;;; -3 -5; 2 2;;;; 3 0; 2 5;;; -4 -2; -4 1;;;; -2 4; -5 1;;; -3 5; 3 -3]


"
backwards compatibility function to extract payoff in format compatible with output of mixed_extension.py
player first, and start indexing at 0, not at 1
Syntax
u_pure[a1, a2, a3, i] == u_func(i, (a1-1, a2-1, a3-1)) for ai in [1, 2]
u_pure[a1+1, a2+1, a3+1, i] == u_func(i, (a1, a2, a3)) for ai in [0, 1]
"
function u_func(i, a)
    a1, a2, a3 = a
    u_pure[a1+1, a2+1, a3+1, i]
end

function see_payoffs()
    players = 1:N
    # Pure strategies
    pures_play = [ 1:A[i] for i in players ]
    pures = [(a1, a2, a3) for a1 in pures_play[1], a2 in pures_play[2], a3 in pures_play[3]]
    j = 1
    for i in players
        for a in pures
            # println("$j: Player $i, pure $([ai-1 for ai in a]), utility =  $(u_pure[a..., i])")
            println(" \\pay_{$i}$([ai-1 for ai in a]) & = $(u_pure[a..., i])\\\\")
            j = j+1

            a1, a2, a3 = a
            # @assert u_pure[a1, a2, a3, i] == u_func(i, (a1-1, a2-1, a3-1))
        end
    end
    # println(u_pure)
end



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




# Reduced replicator dynamics
function RD(x, p, t)
    # p and t are dummy for ODE solver
    # if needed put here global minus sign
    x .* ( ones(3) .- x) .* v(x)
end


#### reduced (pulled back) prox map
function prox(x, step)
    num = x .* exp.( step * v(x) )
    println(num)
    den = ones(3) + num - x
    num ./ den
end

# Euler replicator update
function euler_RD_dynamics(x0, num_iterations, step)
    x = x0
    trajectory = [x0]

    for t in 1:num_iterations
        x = x + step * RD(x, 0, 0)
        push!(trajectory, x)
    end

    x1 = [x[1] for x in trajectory]
    x2 = [x[2] for x in trajectory]
    x3 = [x[3] for x in trajectory]
    return x1, x2, x3
end

function mirror_descent(x0, num_iterations, step)
    x = x0
    trajectory = [x0]

    for t in 1:num_iterations
        x = prox(x, step)
        push!(trajectory, x)
    end

    x1 = [x[1] for x in trajectory]
    x2 = [x[2] for x in trajectory]
    x3 = [x[3] for x in trajectory]
    return x1, x2, x3
end

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
    #mkdir("experiments/$experiment_name")
    # open("experiments/$experiment_name/$experiment_name.txt","a") do io
    open("experiments/H-P-icml/$experiment_name.txt","a") do io
        println(io,"u_pure=",u_pure)
        println(io,"comment=",comment)
    end
    # savefig("experiments/$experiment_name/$experiment_name.pdf")
    savefig("experiments/H-P-icml/$experiment_name.pdf")
end

#############################

x_test = [0.1, 0.6, 0.3]


function run_experiment()
    # euler_primal_space_traj = euler_RD_dynamics(x_test, 2000, 0.005)
    # mirror_descent_traj = mirror_descent(x_test, 2000, 0.005)
    ode_traj = ode_RD_dynamics(x_test, (0, 1000))

    ode1, ode2, ode3 = ode_traj
    
    # plot_title = "(Reduced) RD on 2x2x2 harmonic game"
    plot_title = "RD on 2x2x2 game $pot_param pot + $(round((1-pot_param), sigdigits=2)) harm"
    plot(ode_traj..., label = "RD", xlims=(0, 1), ylims=(0,1), zlims = (0,1), linestyle=:solid, lw = 1, title = plot_title, size=(800,800), margin = -100Plots.mm)
    # plot!(euler_primal_space_traj..., label = "Euler primal")
    # plot!(mirror_descent_traj..., label = "Mirror descent")

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
    scatter!( [  [a] for a in x_test  ]..., label = "Initial point")
    
    # save_experiment("H-P-$pot_param", "")
    
end


###
see_payoffs()
run_experiment()

