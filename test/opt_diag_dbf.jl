using DBF
using PauliOperators
using Test

function run()
    N = 10 
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, -1, -1, -1, z=.1)
    DBF.coeff_clip!(H)
    H0 = deepcopy(H)
   
    g,a = DBF.get_1d_neel_state_sequence(N) 
   
    for (gi, ai) in zip(g,a)
        H = evolve(H, gi, ai)
    end
    
    ψ = Ket{N}(Int128(0))
    display(ψ)
    e0 = expectation_value(H,ψ)
    
    @show e1 = minimum(real(eigvals(Matrix(H))))
   
    @printf(" E0 = %12.8f\n", e0)

    display(ψ)
    e0,e2 = DBF.pt2(H,ψ)
    @printf(" E0 = %12.8f E2 = %12.8f\n", e0, e0+e2)
    
    e0, evar, x, basis = DBF.fois_ci(H, ψ, thresh=1e-4, tol=1e-12, verbose=0)
    
    @time res = DBF.dbf_diag(H,  
                            verbose=1, 
                            max_iter=5, conv_thresh=1e-3, 
                            evolve_coeff_thresh=1e-4,
                            max_rots_per_grad=20)
    

    H = res[1]
    kidx = argmin([real(expectation_value(H,Ket{N}(ψi))) for ψi in 1:2^N])
    ψ = Ket{N}(kidx)
    display(ψ)
    e0, e2 = DBF.pt2(H,ψ)
    @printf(" E0 = %12.8f E2 = %12.8f\n", e0, e0+e2)
    
    e0, ecepa, x, basis = DBF.cepa(H, ψ, thresh=1e-4, tol=1e-12, verbose=0)
    # @printf(" E0 = %12.8f E(CEPA) = %12.8f\n", e0, ecepa)
    
    e0, evar, x, basis = DBF.fois_ci(H, ψ, thresh=1e-4, tol=1e-12, verbose=0)
    # @printf(" E0 = %12.8f E(VAR) =  %12.8f\n", e0, evar[1])
    
end

run()