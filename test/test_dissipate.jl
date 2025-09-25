using Test
using Random

# @testset "theta_opt" begin
function test()
    N = 4 
    H = DBF.heisenberg_1D(N,1,2,3,z=.1)
    O = PauliSum(Pauli(N,Z=[1]))
    ψ = Ket{N}(0)
    display(O)
    
    G = Vector{PauliBasis{N}}([])
    h = Vector{Float64}([])
    for (p,c) in H
        push!(G, p)
        push!(h, c)
    end

    dt = .1
    nsteps = 100
   
    O0 = deepcopy(O)
    # Oval = expectation_value(O,ψ)
    Oval = inner_product(O0,O) 
    @printf(" %12.8f %12.8f %8i %12.8f\n", 0, Oval, length(O), norm(O))
    
    for ti in 1:nsteps
        for (gi,hi) in zip(G,h)
            O = evolve(O,gi,hi*dt)
            DBF.coeff_clip!(O,thresh=1e-4)
        end
        DBF.dissipate!(O,2,1*dt)
        # Oval = expectation_value(O,ψ)
        Oval = inner_product(O0,O) 
        @printf(" %12.8f %12.8f %8i %12.8f\n", ti*dt, Oval, length(O), norm(O))
    end
end

test()