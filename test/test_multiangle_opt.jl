using Test
using Random
using Printf
using PauliOperators
using LinearAlgebra
using DBF
using BenchmarkTools
using Optim

function cost_function(Hin, generators, angles, ψ)

    Hxz = pack_x_z(Hin)

    length(generators) == length(angles) || throw(DimensionMismatchi)

    ψt = deepcopy(ψ)

    for (gi, θi) in zip(generators, angles)
        evolve!(ψt, gi, θi)
    end


    e = expectation_value(Hxz, ψt)

    return real(e)
end
function gradient(Hin, generators, angles, ψ)

    Hxz = pack_x_z(Hin)

    length(generators) == length(angles) || throw(DimensionMismatchi)

    ψt = deepcopy(ψ)

    for (gi, θi) in zip(generators, angles)
        evolve!(ψt, gi, θi)
    end


    e = expectation_value(Hxz, ψt)

  
    # Now compute gradient
    σt = DBF.matvec(Hxz, ψt)
    gt = zeros(length(angles))
    op_idx = 1
    for (gi, θi) in zip(reverse(generators), reverse(angles))
        gt[op_idx] = imag(matrix_element(σt, gi, ψt))
        evolve!(ψt, gi, -θi)
        evolve!(σt, gi, -θi)
        op_idx += 1
    end

    return gt
end

@testset "test_opt" begin
# function test()
    Random.seed!(2)

    N = 6 
    H = rand(PauliSum{N}, n_paulis=100)
    H = H + H'
    
    generators = Vector{PauliBasis{N}}([])
    angles = Vector{Float64}([])
    for i in 1:20 
        push!(generators, rand(PauliBasis{N}))
        push!(angles, rand()-.5)
    end

    ket = KetSum(Ket{N}(0), T=ComplexF64)

    # @show typeof(generators), typeof(angles), typeof(ket) 


    Ht, _ = evolve(H, generators, angles, thresh=-1)
    @show e = expectation_value(Ht, ket)

    @show e = cost_function(H, reverse(generators), reverse(angles), ket)
    @show g = gradient(H, reverse(generators), reverse(angles), ket)

    # Check with numerical gradients
    h = .001
    gnumerical = zeros(length(angles))
    for i in 1:length(angles)
        a1 = deepcopy(angles)
        a2 = deepcopy(angles)
        a1[i] += h
        a2[i] -= h
        e1 = cost_function(H, reverse(generators), reverse(a1), ket)
        e2 = cost_function(H, reverse(generators), reverse(a2), ket)
        gnumerical[i] = real(e1 - e2)/(2*h)
    end

    gnumerical = real(gnumerical)
    for (i,j) in zip(g, gnumerical)
        @printf(" %12.8f %12.8f %12.8f\n", i, j, j - i)
        @test(abs(i-j) < 1e-5)
    end

    f(θ) = cost_function(H, reverse(generators), reverse(θ), ket)
    g(θ) = gradient(H, reverse(generators), reverse(θ), ket)

    function fg!(F, G, x)
        Hxz = pack_x_z(H)
        length(generators) == length(x) || throw(DimensionMismatch)
        ψt = deepcopy(ket)

        for (gi, θi) in zip(generators, x)
            evolve!(ψt, gi, θi)
        end
        e = expectation_value(Hxz, ψt)

        if G !== nothing
            length(G) == length(x) || throw(DimensionMismatch)
            σt = DBF.matvec(Hxz, ψt)
            gt = zeros(length(x))
            op_idx = 1
            for (gi, θi) in zip(reverse(generators), reverse(x))
                gt[op_idx] = imag(matrix_element(σt, gi, ψt))
                evolve!(ψt, gi, -θi)
                evolve!(σt, gi, -θi)
                op_idx += 1
            end
            G .= gt
        end
        if F !== nothing
            return real(e) 
        end
    end

    # return
    x0 = zeros(length(generators))
    options = Optim.Options(
        x_reltol = 1e-12, # A tight relative tolerance for changes in the solution vector
        f_reltol = 1e-12, # A tight relative tolerance for changes in the objective function value
        g_tol = 1e-10,    # A tighter absolute tolerance for the gradient
        store_trace=true,
    )
    # result = optimize(Optim.only_fg!(fg!), zeros(length(generators)), LBFGS())
    # θ = result.minimizer
    # f_min = result.minimum
    # # result = optimize(f, g, LBFGS())
    
    result = optimize(f, (G, x) -> G .= g(x), x0, LBFGS(), options)
    θ = result.minimizer
    f_min = result.minimum

    for t in Optim.trace(result)
        @printf(" %4i %12.8f %12.8f\n", t.iteration, t.value, t.g_norm)
    end


    if Optim.iteration_limit_reached(result)
        @show Optim.abs_tol(result), Optim.rel_tol(result)
        @warn " minimization failed"
    end

    display(eigvals(Matrix(H)))
end


# test()