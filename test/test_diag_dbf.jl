
function test_diag_dbf(;max_iter=100)
    N = 3 
    Random.seed!(2)
    H = DBF.heisenberg_1D(N, 1, 2, 3, z=.1)
    DBF.coeff_clip!(H)

    println(" Original H:")
    display(H)
    evals1 = eigvals(Matrix(H))
    evals2 = eigvals(Matrix(diag(H)))
    @printf(" ||H||  = %12.8f\n", norm(H))
    @printf(" ||Hd|| = %12.8f\n", norm(diag(H)))
    @printf(" ||Ho|| = %12.8f\n", norm(DBF.offdiag(H)))
    H, gi, θi = dbf_diag(H, max_iter=1000, conv_thresh=1e-7, evolve_coeff_thresh=1e-3)
    
    # println(" New H:")
    # display(H)
    evals3 = eigvals(Matrix(H))
    evals4 = eigvals(Matrix(diag(H)))
    for i in 1:2^N
        @printf(" %3i %12.8f %12.8f %12.8f %12.8f\n", i, evals1[i], evals2[i], evals3[i], evals4[i])
    end
end

test_diag_dbf()