using DBF
using Plots
using DifferentialEquations

"""
    dissipate!(O::PauliSum, Li::PauliBasis, γ::Real, dt::Real)

Form derived by Chinmay:
O(t) -> O  + γ dt exp{-2 γ dt} (Li O Li - O)

This assumes the jump operator, Li, is a single pauli. 
"""
function dissipate!(O::PauliSum, Li::PauliBasis, γ::Real, dt::Real)
    exp_term = exp(-2 * γ * dt)
    for (p,c) in O
        
        # [p,Li] == 0, then we do nothing
        !PauliOperators.commute(p,Li) || continue

        O[p] = c * ( 1 - 2 * γ * dt * exp_term)
    end
    return O
end

function run()

    # N = 49 
    N = 4 
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, -1, -2, -3, z=.1)
    # H = DBF.heisenberg_2D(2, 2, -1, -1, -1, z=.1)
    # H = DBF.heisenberg_2D(7, 7, -0, -0, -1, x=.1)
    # H = DBF.heisenberg_2D(N, 1, -1, -2, -3, x=.1)
    DBF.coeff_clip!(H)

    H_trotter = Vector{Tuple{PauliBasis{N}, Float64}}([(k,v) for (k,v) in H])

    # create measurement operators
    γ = .1
    L = [(PauliBasis(Pauli(N, Z=[i])), γ) for i in 1:N]
    println(" Original H:")
    # display(H)
    display("H_trotter:")
    [@printf(" %s %12.8f\n", string(li[1]), li[2]) for li in H_trotter]
    display("L_trotter:")
    [@printf(" %s %12.8f\n", string(li[1]), li[2]) for li in L]
    # evals = eigvals(Hmat)

    # Initial operator:
    O0 = PauliSum(Pauli(N, Z=[1], X=[2]))
    Ot = deepcopy(O0) 
    n_steps = 100
    dt = .01
    

    Hmat = Matrix(H)
    Umat = exp(-1im * Hmat * dt)
    Omat = Matrix(Ot)
    O0mat = Matrix(Ot)
    Lmat = [(Matrix(l[1]), l[2]) for l in L]


    tplot = [i*dt for i in 0:n_steps]
    p_plot = zeros(n_steps+1)
    m_plot = zeros(n_steps+1)

    @printf(" %12s %12s %12s\n", "t", "Pauli", "Matrix")
    @printf(" %12.8f %12.8f %12.8f\n", 0, inner_product(O0,Ot), real(tr(O0mat*Omat))/2^N)
    
    p_plot[1] = inner_product(O0,Ot)
    m_plot[1] = real(tr(O0mat*Omat))/2^N
    
    for step_i in 1:n_steps

        # evolve by H
        for (ok, hk) in H_trotter
            # O(t+dt) = exp(i hi ok dt) O(t) exp(i hi ok dt)
            # θ = 2 * hi * dt 
            DBF.evolve!(Ot, ok, 2*hk*dt)
        end
        # Dissipate
        for (Lk, γk) in L 
            dissipate!(Ot, Lk, γk, dt)
        end
    
        # Exact:
        Omat += dt * 1im*(Hmat*Omat - Omat*Hmat)
        # Omat = Umat'*Omat*Umat
        # Dissipate
        for (Lk, γk) in Lmat 
            Omat += dt * γk * (Lk' * Omat * Lk - 1/2 * Lk'*Lk*Omat - 1/2 * Omat*Lk'*Lk) 
        end
        
        @printf(" %12.8f %12.8f %12.8f\n", step_i*dt, inner_product(O0,Ot), real(tr(O0mat*Omat))/2^N)
        Omat = Omat / norm(Omat)

        # store at step_i+1 because tplot[1] corresponds to time 0
        p_plot[step_i+1] = inner_product(O0,Ot)
        m_plot[step_i+1] = real(tr(O0mat*Omat))/2^N
        
    end

    function rhs(o,p,t)
        # Unitary
        dto = 1im*(Hmat*o - o*Hmat)
        # Dissipation
        for (Lk, γk) in Lmat 
            dto += γk * (Lk' * o * Lk - 1/2 * Lk'*Lk*o - 1/2 * o*Lk'*Lk) 
        end
        # Backaction
        return dto 
    end

    # prob = ODEProblem(rhs, O0mat, 0:dt:n_steps*dt)
    prob = ODEProblem(rhs, O0mat, (0,n_steps*dt))
    sol = solve(prob, saveat=dt)
    exact = [real(tr(oi*O0mat))/2^N for oi in sol.u]
    # plot(sol)
    

    # Plot results (requires Plots.jl)
    plt = plot(tplot, p_plot, label="Pauli inner product", xlabel="t", ylabel="value", legend=:topright)
    # plot!(plt, tplot, m_plot, label="Matrix fidelity")
    plot!(plt, sol.t, exact, label="ODE", markerstyle="o")
    ylims!(-1,1)
    savefig(plt, "cont_measure_plot.pdf")

    @show norm(exact - p_plot)
    # println(" Now exact:")
    # Hmat = Matrix(H)
    # Umat = exp(-1im * Hmat * dt)
    # Omat = Matrix(Ot)
    # O0mat = Matrix(Ot)
    # @printf(" %12.8f %12.8f\n", 0, real(tr(O0mat*Omat))/2^N)
    # for step_i in 1:n_steps

    #     # evolve by H

    #     Omat = Umat'*Omat*Umat

        
    #     @printf(" %12.8f %12.8f\n", step_i*dt, real(tr(O0mat*Omat))/2^N)
        
    # end
end



run()