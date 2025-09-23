
function test_diag_dbf(;max_iter=100)
    N = 2
    Random.seed!(2)
    H = rand(PauliSum{N}, n_paulis=100)
    H += H'

    println(" Original H:")
    display(H)

    H, gi, θi = DBF.dbf_diag(H, max_iter=100, conv_thresh=1e-8)
    
    println(" New H:")
    display(H)
end

test_diag_dbf()