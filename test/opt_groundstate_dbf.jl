using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test


function run()
    N = 6 
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, -1, -1, -1, x=.1)
    # H = DBF.heisenberg_2D(2, 2, -1, -1, -1, z=.1)
    # H = DBF.heisenberg_2D(7, 7, -0, -0, -1, x=.1)
    DBF.coeff_clip!(H)
   
    # Transform H to make |000> the most stable bitstring
    for i in 1:N
        if i%2 == 0
            H = Pauli(N, X=[i]) * H * Pauli(N, X=[i])
        end
    end 
    
    H0 = deepcopy(H)
    # display(H)
    
    # ψ = Ket([i%2 for i in 1:N])
    ψ = Ket([0 for i in 1:N])
    # ψ += Ket([i%2 for i in 0:N-1])
    # PauliOperators.scale!(ψ, 1/norm(ψ))

    # kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 1:2^N])
    # ψ = Ket{N}(kidx)
    display(ψ)
    e0 = expectation_value(H,ψ)
    
    @show e1 = minimum(real(eigvals(Matrix(H))))
   
    @printf(" E0 = %12.8f\n", e0)

    @show norm(H)
    
    @time H, g, θ = DBF.dbf_groundstate(H, ψ, n_body=1, 
                                verbose=1, 
                                max_iter=120, conv_thresh=1e-3, 
                                evolve_coeff_thresh=1e-3,
                                # evolve_weight_thresh=5,
                                grad_coeff_thresh=1e-3,
                                # grad_weight_thresh=2,
                                search_n_top=100)
    # @show DBF.get_weight_counts(H)
    # @show DBF.get_weight_probs(H)
    
    @time H, g2, θ2 = DBF.dbf_groundstate(H, ψ, n_body=1,
                                verbose=1, 
                                max_iter=120, conv_thresh=1e-3, 
                                evolve_coeff_thresh=1e-3,
                                # evolve_weight_thresh=8,
                                grad_coeff_thresh=1e-3,
                                # grad_weight_thresh=3,
                                search_n_top=200)
    g = vcat(g,g2)
    θ = vcat(θ,θ2)
    # @show DBF.get_weight_counts(H)
    # @show DBF.get_weight_probs(H)
    # @time H, g2, θ2 = DBF.dbf_groundstate(H, ψ, n_body=1,
    #                             verbose=1, 
    #                             max_iter=120, conv_thresh=1e-3, 
    #                             evolve_coeff_thresh=1e-3,
    #                             # evolve_weight_thresh=8,
    #                             grad_coeff_thresh=1e-3,
    #                             search_n_top=10000)
    # g = vcat(g,g2)
    # θ = vcat(θ,θ2)
    # # @show DBF.get_weight_counts(H)
    # # @show DBF.get_weight_probs(H)
    

    Ht = deepcopy(H0)
    err = 0
    ecurr = expectation_value(Ht,ψ)
    println("\n Now rerun with higher accuracy:")
    println("    # of rotations: ", length(θ))
    @printf(" Initial energy: %12.8f %8i\n", ecurr, length(Ht))
    for (gi,θi) in zip(g,θ)
            
        Ht = DBF.evolve(Ht, gi, θi)
        
        e1 = expectation_value(Ht,ψ)
        DBF.coeff_clip!(Ht, thresh=1e-5)
        e2 = expectation_value(Ht,ψ)

        err += e2 - e1
    end    
    ecurr = expectation_value(Ht,ψ)
    @printf(" ecurr %12.8f err %12.8f %8i\n", ecurr, err, length(Ht))



    println("\n Now reroptimize with higher accuracy:")
    @show length(θ)
    Ht = deepcopy(H0)
    err = 0
    ecurr = expectation_value(Ht,ψ)
    @printf(" Initial energy: %12.8f %8i\n", ecurr, length(Ht))
    for (i,gi) in enumerate(g)
            
        θj, costi = DBF.optimize_theta_expval(Ht, gi, ψ, verbose=0)
        Ht = DBF.evolve(Ht, gi, θj)
        θ[i] = θj
        
        e1 = expectation_value(Ht,ψ)
        DBF.coeff_clip!(Ht, thresh=1e-5)
        e2 = expectation_value(Ht,ψ)

        err += e2 - e1
    end    
    ecurr = expectation_value(Ht,ψ)
    @printf(" ecurr %12.8f err %12.8f %8i\n", ecurr, err, length(Ht))
   
    #
    println("\n Now compute GS properties: <GS|ZZ|GS>")
    Ot = PauliSum(Pauli(N,X=[1,3]))
    O0 = deepcopy(Ot)
    println("    # of rotations: ", length(θ))
    err = 0
    @printf(" Initial O: %12.8f %8i\n", expectation_value(Ot,ψ), length(Ot))
    for (gi,θi) in zip(g,θ)
            
        Ot = DBF.evolve(Ot, gi, θi)
        
        e1 = expectation_value(Ot,ψ)
        DBF.coeff_clip!(Ot, thresh=1e-5)
        e2 = expectation_value(Ot,ψ)

        err += e2 - e1
        # ot = expectation_value(Ot,ψ)
        # @printf(" %12.8f %12.8f\n",ot, ot-err)
       
    end    
    
    @printf(" ocurr %12.8f err %12.8f %8i\n", expectation_value(Ot,ψ), err, length(Ot))
    e,v = eigen(Matrix(H0))
    v = v[:,1]
    
    @show v'*Matrix(H0)*v
    @show v'*Matrix(O0)*v

    # Build H in subspace
    # basis_dict = DBF.add_single_excitations(ψ)
   
    
    basis_dict = Ht*ψ
    @show length(basis_dict)
    # DBF.coeff_clip!(basis_dict, thresh=1e-3)
    @show length(basis_dict)
    basis = Vector{Ket{N}}([ψ])
    for (k,c) in basis_dict
        k != ψ || continue 
        push!(basis, k)
    end
    H = Matrix(Ht,basis)
    display(eigvals(H)) 
    display(basis[1])
    v = H[1,2:end]
    d = e0 .- diag(H)[2:end]
    d = 1 ./ d
    @show v' * (d .* v)
    e2 = 0
    e0 = H[1,1]
    for i in 2:length(basis)
        basis[i] != ψ || error(" ψ == i")
        e2 += H[1,i]*H[i,1] / (e0 - H[i,i])
    end
    @show e0, e2


    println("\n Compute PT2 correction")
    e0, e2 = DBF.pt2(Ht, ψ)
    @printf(" E0 = %12.8f E2 = %12.8f EPT2 = %12.8f \n", e0, e2, e0+e2)

end


run()