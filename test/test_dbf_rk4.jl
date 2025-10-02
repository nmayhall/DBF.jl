using PauliOperators
using LinearAlgebra
using Printf

# include("rk4.jl")
# include("helpers.jl")


function low_weight_mult(ps1::PauliSum{N,T}, ps2::PauliSum{N,T}, w) where {N,T}
    out = PauliSum(N, T)
    for (op1, coeff1) in ps1 
        for (op2, coeff2) in ps2
            prod = Pauli(op1) * Pauli(op2)

            c = coeff(prod)
            prod = PauliBasis(prod)
            
            # make sure weight is small
            weight(prod) <= w || continue
            
            curr = get(out, prod, 0.0)  
            out[prod] = curr + c*coeff1*coeff2 

        end
    end
    return out 
end

"""
    double_commutator(O::PauliSum{N}) where N

Compute the double commutator: [O, [O, diag(O)]]

This is the right-hand side for diagonalization DBF.
"""
function double_commutator(O::PauliSum{N}) where N
    Od = diag(O)
   
    # First commutator: [O, diag(O)]
    comm1 = O * Od - Od * O
    
    # Second commutator: [O, [O, diag(O)]]
    comm2 = O * comm1 - comm1 * O
    
    return comm2
end


function double_commutator_low_weight(O::PauliSum{N}, w) where N
    Od = diag(O)
   
    o1 = low_weight_mult(O,Od,w) - low_weight_mult(Od,O,w)
    o2 = low_weight_mult(O,o1,w) - low_weight_mult(o1,O,w)
    
    return o2 
end


"""
    dbf_diag_rk4(Oin::PauliSum{N,T};
                 t_end=10.0,
                 dt=0.01,
                 truncate_coeff=1e-8,
                 truncate_weight=nothing,
                 verbose=1,
                 conv_thresh=1e-6,
                 check_interval=10) where {N,T}

Solve the diagonalization DBF using RK4 integration:
    dO/dt = [O, [O, diag(O)]]

# Arguments
- `Oin`: Initial operator (PauliSum)
- `t_end`: Final integration time
- `dt`: Time step for RK4
- `truncate_coeff`: Truncate coefficients smaller than this
- `truncate_weight`: Truncate terms with weight greater than this
- `verbose`: Verbosity level (0=quiet, 1=progress, 2=detailed)
- `conv_thresh`: Stop if ||offdiag(O)|| < conv_thresh
- `check_interval`: Check convergence every this many steps

# Returns
- `O_final`: Diagonalized operator
- `t_history`: Vector of time points
- `O_history`: Vector of operators at each time
- `metrics`: Dictionary of diagnostic metrics

# Example
```julia
N = 4
H = heisenberg_1D(N, 1.0, 1.0, 1.0)
O_final, t, O_hist, metrics = dbf_diag_rk4(H, 
                                           t_end=5.0, 
                                           dt=0.01,
                                           truncate_coeff=1e-6,
                                           truncate_weight=3)
```
"""
function dbf_diag_rk4(Oin::PauliSum{N,T};
                      t_end=10.0,
                      dt=0.01,
                      truncate_coeff=1e-8,
                      truncate_weight=nothing,
                      verbose=2,
                      conv_thresh=1e-6,
                      check_interval=10) where {N,T}
    
    # Initialize storage for metrics
    metrics = Dict{String, Vector{Float64}}(
        "time" => Float64[],
        "norm_total" => Float64[],
        "norm_diag" => Float64[],
        "norm_offdiag" => Float64[],
        "length" => Float64[]
    )
    
    # Step counter for convergence checking
    step_count = Ref(0)
    converged = Ref(false)
    
    # Right-hand side function
    function rhs(t, O)
        return double_commutator_low_weight(O,truncate_weight)
        # return double_commutator(O)
    end
    
    # Callback to check convergence and print progress
    function callback(t, O)
        step_count[] += 1
        
        # Only check every check_interval steps
        if mod(step_count[], check_interval) == 0
            norm_total = norm(O)
            norm_d = norm(diag(O))
            norm_od = norm(offdiag(O))
            len_O = length(O)
            
            # Store metrics
            push!(metrics["time"], t)
            push!(metrics["norm_total"], norm_total)
            push!(metrics["norm_diag"], norm_d)
            push!(metrics["norm_offdiag"], norm_od)
            push!(metrics["length"], len_O)
            
            # Print progress
            if verbose >= 1
                @printf("t=%8.4f |O|=%10.6f |diag|=%10.6f |offdiag|=%10.6f len=%6i\n",
                        t, norm_total, norm_d, norm_od, len_O)
            end
            
            # Check convergence
            if norm_od < conv_thresh
                if verbose >= 1
                    @show length(O)
                    @printf("Converged at t=%.4f\n", t)
                end
                converged[] = true
                return true  # Stop integration
            end
        end
       
        flush(stdout)
        return false  # Continue
    end
    
    # Print header
    if verbose >= 1
        @printf("\n=== DBF Diagonalization with RK4 ===\n")
        @printf("Initial: |O|=%10.6f |diag|=%10.6f |offdiag|=%10.6f len=%6i\n",
                norm(Oin), norm(diag(Oin)), norm(offdiag(Oin)), length(Oin))
        @printf("\nIntegrating...\n")
    end
    
    # Integrate
    t_history, O_history = DBF.rk4_integrate(
        rhs, Oin, (0.0, t_end), dt,
        callback=callback,
        truncate_coeff=truncate_coeff,
        truncate_weight=truncate_weight
    )
    
    O_final = O_history[end]
    
    # Final summary
    if verbose >= 1
        @printf("\n=== Final Results ===\n")
        @printf("Final time: t=%.4f\n", t_history[end])
        @printf("Converged: %s\n", converged[] ? "Yes" : "No")
        @printf("Final: |O|=%10.6f |diag|=%10.6f |offdiag|=%10.6f len=%6i\n",
                norm(O_final), norm(diag(O_final)), norm(offdiag(O_final)), length(O_final))
        @printf("Diagonalization ratio: %.6f\n", norm(diag(O_final)) / norm(O_final))
    end
    
    return O_final, t_history, O_history, metrics
end



"""
    dbf_diag_rk4_adaptive(Oin::PauliSum{N,T};
                          t_end=10.0,
                          dt_init=0.1,
                          tol=1e-6,
                          truncate_coeff=1e-8,
                          truncate_weight=nothing,
                          verbose=1,
                          conv_thresh=1e-6,
                          check_interval=10) where {N,T}

Adaptive version of DBF diagonalization with RK4.

Same as `dbf_diag_rk4` but uses adaptive time stepping.
"""
function dbf_diag_rk4_adaptive(Oin::PauliSum{N,T};
                               t_end=10.0,
                               dt_init=0.1,
                               tol=1e-6,
                               dt_min=1e-8,
                               dt_max=1.0,
                               truncate_coeff=1e-8,
                               truncate_weight=nothing,
                               verbose=1,
                               conv_thresh=1e-6,
                               check_interval=10) where {N,T}
    
    # Initialize storage for metrics
    metrics = Dict{String, Vector{Float64}}(
        "time" => Float64[],
        "norm_total" => Float64[],
        "norm_diag" => Float64[],
        "norm_offdiag" => Float64[],
        "length" => Float64[]
    )
    
    step_count = Ref(0)
    converged = Ref(false)
    
    function rhs(t, O)
        # return double_commutator(O)
        return double_commutator_low_weight(O,truncate_weight)
    end
    
    function callback(t, O)
        step_count[] += 1
        
        if mod(step_count[], check_interval) == 0
            norm_total = norm(O)
            norm_d = norm(diag(O))
            norm_od = norm(offdiag(O))
            len_O = length(O)
            
            push!(metrics["time"], t)
            push!(metrics["norm_total"], norm_total)
            push!(metrics["norm_diag"], norm_d)
            push!(metrics["norm_offdiag"], norm_od)
            push!(metrics["length"], len_O)
            
            if verbose >= 1
                @printf("t=%8.4f |O|=%10.6f |diag|=%10.6f |offdiag|=%10.6f len=%6i\n",
                        t, norm_total, norm_d, norm_od, len_O)
            end
            
            if norm_od < conv_thresh
                if verbose >= 1
                    @printf("Converged at t=%.4f\n", t)
                end
                converged[] = true
                return true
            end
        end
        
        return false
    end
    
    if verbose >= 1
        @printf("\n=== DBF Diagonalization with Adaptive RK4 ===\n")
        @printf("Initial: |O|=%10.6f |diag|=%10.6f |offdiag|=%10.6f len=%6i\n",
                norm(Oin), norm(diag(Oin)), norm(offdiag(Oin)), length(Oin))
        @printf("\nIntegrating...\n")
    end
    
    t_history, O_history = rk4_integrate_adaptive(
        rhs, Oin, (0.0, t_end), dt_init,
        tol=tol,
        dt_min=dt_min,
        dt_max=dt_max,
        callback=callback,
        truncate_coeff=truncate_coeff,
        truncate_weight=truncate_weight
    )
    
    O_final = O_history[end]
    
    if verbose >= 1
        @printf("\n=== Final Results ===\n")
        @printf("Final time: t=%.4f\n", t_history[end])
        @printf("Steps taken: %i\n", length(t_history))
        @printf("Converged: %s\n", converged[] ? "Yes" : "No")
        @printf("Final: |O|=%10.6f |diag|=%10.6f |offdiag|=%10.6f len=%6i\n",
                norm(O_final), norm(diag(O_final)), norm(offdiag(O_final)), length(O_final))
        @printf("Diagonalization ratio: %.6f\n", norm(diag(O_final)) / norm(O_final))
    end
    
    return O_final, t_history, O_history, metrics
end


# # Example usage
# if abspath(PROGRAM_FILE) == @__FILE__
    using Random
    
    println("Testing DBF Diagonalization with RK4")
    
    # Create test Hamiltonian
    N = 5 
    Random.seed!(2)
    
    function heisenberg_1D(N, Jx, Jy, Jz; x=0, y=0, z=0)
        H = PauliSum(N, Float64)
        for i in 0:N-1
            H += -2*Jx * Pauli(N, X=[i+1,(i+1)%(N)+1])
            H += -2*Jy * Pauli(N, Y=[i+1,(i+1)%(N)+1])
            H += -2*Jz * Pauli(N, Z=[i+1,(i+1)%(N)+1])
        end 
        for i in 1:N
            H += x * Pauli(N, X=[i])
            H += y * Pauli(N, Y=[i])
            H += z * Pauli(N, Z=[i])
        end 
        return H
    end
    
    H = heisenberg_1D(N, -1.1, -1.2, -1.3, z=0.1)
    
    println("\n--- Test 1: Fixed time step ---")
    O1, t1, hist1, metrics1 = dbf_diag_rk4(H,
                                           t_end=1.0,
                                           dt=0.001,
                                           truncate_coeff=1e-4,
                                           truncate_weight=3,
                                           verbose=2)
    
    println("\n--- Test 2: Adaptive time step ---")
    O2, t2, hist2, metrics2 = dbf_diag_rk4_adaptive(H,
                                                     t_end=2.0,
                                                     dt_init=0.01,
                                                     tol=1e-5,
                                                     truncate_coeff=1e-4,
                                                     truncate_weight=4,
                                                     verbose=2)
    
    # Compare eigenvalues
    println("\n--- Eigenvalue comparison ---")
    evals_orig = sort(real(eigvals(Matrix(H))))
    evals_final1 = sort(real(eigvals(Matrix(O1))))
    # evals_final2 = sort(real(eigvals(Matrix(O2))))
    
    # @printf("%3s %12s %12s %12s\n", "i", "Original", "Fixed", "Adaptive")
    # for i in 1:min(8, 2^N)
    #     @printf("%3i %12.6f %12.6f %12.6f\n", i, evals_orig[i], evals_final1[i], evals_final2[i])
    # end
# end