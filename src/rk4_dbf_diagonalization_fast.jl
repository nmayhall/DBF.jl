using PauliOperators
using LinearAlgebra
using Printf
using DBF
include("rk4.jl")
# include("helpers.jl")
include("truncation_strategies.jl")


"""
    dbf_diag_rk4_fast(Oin::PauliSum{N,T};
                      t_end=10.0,
                      dt=0.01,
                      max_terms=500,
                      coeff_thresh=1e-6,
                      max_weight=nothing,
                      adaptive_truncation=true,
                      verbose=1,
                      conv_thresh=1e-6,
                      check_interval=10) where {N,T}

Fast version of DBF diagonalization using smart truncation strategies.

This is much faster than the naive version because it:
1. Truncates intermediate results during commutator computation
2. Adaptively adjusts truncation based on operator size
3. Uses top-k selection to keep most important terms

# Arguments
- `Oin`: Initial operator
- `t_end`: Final time
- `dt`: Time step
- `max_terms`: Maximum number of Pauli terms to keep
- `coeff_thresh`: Coefficient threshold for truncation
- `max_weight`: Maximum weight of Pauli strings to keep (locality)
- `adaptive_truncation`: Adjust truncation dynamically based on size
- `verbose`: Verbosity level
- `conv_thresh`: Convergence threshold for ||offdiag||
- `check_interval`: Check convergence every this many steps

# Returns
- `O_final`: Diagonalized operator
- `t_history`: Time points
- `O_history`: Operator history
- `metrics`: Diagnostic metrics

# Example
```julia
H = heisenberg_1D(6, 1.0, 1.0, 1.0)
O, t, hist, metrics = dbf_diag_rk4_fast(H,
                                        t_end=5.0,
                                        dt=0.01,
                                        max_terms=300,
                                        max_weight=3)
```
"""
function dbf_diag_rk4_fast(Oin::PauliSum{N,T};
                          t_end=10.0,
                          dt=0.01,
                          max_terms=500,
                          coeff_thresh=1e-6,
                          max_weight=nothing,
                          adaptive_truncation=true,
                          verbose=1,
                          conv_thresh=1e-6,
                          check_interval=10) where {N,T}
    
    metrics = Dict{String, Vector{Float64}}(
        "time" => Float64[],
        "norm_total" => Float64[],
        "norm_diag" => Float64[],
        "norm_offdiag" => Float64[],
        "length" => Float64[],
        "max_terms_used" => Float64[],
        "coeff_thresh_used" => Float64[]
    )
    
    step_count = Ref(0)
    converged = Ref(false)
    
    # Right-hand side with smart truncation
    function rhs(t, O)
        # Adaptive truncation parameters
        if adaptive_truncation
            params = adaptive_truncation_params(O, 
                                               base_max_terms=max_terms,
                                               base_coeff_thresh=coeff_thresh)
            mt = params.max_terms
            ct = params.coeff_thresh
        else
            mt = max_terms
            ct = coeff_thresh
        end
        
        # Compute double commutator with smart truncation
        return smart_double_commutator(O,
                                      max_terms=mt,
                                      coeff_thresh=ct,
                                      max_weight=max_weight)
    end
    
    function callback(t, O)
        step_count[] += 1
        
        if mod(step_count[], check_interval) == 0
            norm_total = norm(O)
            norm_d = norm(diag(O))
            norm_od = norm(offdiag(O))
            len_O = length(O)
            
            # Track truncation params
            if adaptive_truncation
                params = adaptive_truncation_params(O, 
                                                   base_max_terms=max_terms,
                                                   base_coeff_thresh=coeff_thresh)
                mt = params.max_terms
                ct = params.coeff_thresh
            else
                mt = max_terms
                ct = coeff_thresh
            end
            
            push!(metrics["time"], t)
            push!(metrics["norm_total"], norm_total)
            push!(metrics["norm_diag"], norm_d)
            push!(metrics["norm_offdiag"], norm_od)
            push!(metrics["length"], len_O)
            push!(metrics["max_terms_used"], mt)
            push!(metrics["coeff_thresh_used"], ct)
            
            if verbose >= 1
                @printf("t=%8.4f |O|=%10.6f |od|=%10.6f len=%5i max_t=%5i thresh=%.1e\n",
                        t, norm_total, norm_od, len_O, mt, ct)
            end
            
            if norm_od < conv_thresh
                if verbose >= 1
                    @printf("✓ Converged at t=%.4f\n", t)
                end
                converged[] = true
                return true
            end
        end
        
        return false
    end
    
    if verbose >= 1
        @printf("\n=== Fast DBF Diagonalization ===\n")
        @printf("Settings: max_terms=%i, coeff_thresh=%.1e, max_weight=%s\n",
                max_terms, coeff_thresh, max_weight === nothing ? "none" : string(max_weight))
        @printf("Adaptive truncation: %s\n", adaptive_truncation)
        @printf("\nInitial: |O|=%10.6f |diag|=%10.6f |offdiag|=%10.6f len=%6i\n",
                norm(Oin), norm(diag(Oin)), norm(offdiag(Oin)), length(Oin))
        @printf("\nIntegrating...\n")
    end
   
    # function rhs(t,O)
    #     tmp = O*diag(O)-diag(O)*O
    #     return O*tmp-tmp*O
    # end
    # Integrate with NO additional truncation in rk4_integrate
    # (we handle it inside the RHS function)
    t_history, O_history = rk4_integrate(
        rhs, Oin, (0.0, t_end), dt,
        callback=callback,
        truncate_coeff=nothing,  # Handled in RHS
        truncate_weight=nothing  # Handled in RHS
    )
    
    O_final = O_history[end]
    
    if verbose >= 1
        @printf("\n=== Final Results ===\n")
        @printf("Final time: t=%.4f (%.0f steps)\n", t_history[end], length(t_history))
        @printf("Converged: %s\n", converged[] ? "Yes" : "No")
        @printf("Final: |O|=%10.6f |diag|=%10.6f |offdiag|=%10.6f len=%6i\n",
                norm(O_final), norm(diag(O_final)), norm(offdiag(O_final)), length(O_final))
        @printf("Diagonalization: %.2f%%\n", 100 * norm(diag(O_final)) / norm(O_final))
    end
    
    return O_final, t_history, O_history, metrics
end


"""
    compare_truncation_strategies(H::PauliSum{N};
                                  t_end=2.0,
                                  dt=0.01) where N

Compare different truncation strategies to see trade-offs.
"""
function compare_truncation_strategies(H::PauliSum{N};
                                      t_end=2.0,
                                      dt=0.01) where N
    
    println("\n" * "="^70)
    println("COMPARING TRUNCATION STRATEGIES")
    println("="^70)
    
    strategies = [
        ("Aggressive", (max_terms=200, coeff_thresh=1e-4, max_weight=3)),
        ("Moderate", (max_terms=500, coeff_thresh=1e-6, max_weight=4)),
        ("Lenient", (max_terms=1000, coeff_thresh=1e-8, max_weight=5)),
        ("Adaptive (base=500)", (max_terms=500, coeff_thresh=1e-6, max_weight=4, adaptive_truncation=true))
    ]
    
    results = []
    
    for (name, params) in strategies
        println("\n--- Strategy: $name ---")
        println("Parameters: $params")
        
        try
            @time O_final, t_hist, O_hist, metrics = dbf_diag_rk4_fast(
                H, t_end=t_end, dt=dt, verbose=0; params...
            )
            
            final_norm_od = norm(offdiag(O_final))
            final_len = length(O_final)
            diag_ratio = norm(diag(O_final)) / norm(O_final)
            
            # Check eigenvalue preservation
            evals_orig = eigvals(Matrix(H))
            evals_final = eigvals(Matrix(O_final))
            eval_error = norm(sort(real(evals_orig)) - sort(real(evals_final)))
            
            push!(results, (name=name, 
                          off_diag=final_norm_od,
                          length=final_len,
                          diag_ratio=diag_ratio,
                          eval_error=eval_error))
            
            @printf("✓ Final: |od|=%.6f len=%i diag=%.2f%% eval_err=%.2e\n",
                    final_norm_od, final_len, 100*diag_ratio, eval_error)
        catch e
            println("✗ Failed: $e")
            push!(results, (name=name, off_diag=NaN, length=0, diag_ratio=NaN, eval_error=NaN))
        end
    end
    
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    @printf("%-25s %12s %8s %10s %12s\n", "Strategy", "|offdiag|", "Length", "Diag %", "Eval Error")
    println("-"^70)
    for r in results
        @printf("%-25s %12.6f %8i %9.1f%% %12.2e\n",
                r.name, r.off_diag, r.length, 100*r.diag_ratio, r.eval_error)
    end
    
    return results
end


# Test if run directly
# if abspath(PROGRAM_FILE) == @__FILE__
    using Random
    
    println("Testing Fast DBF Diagonalization")
    
    Random.seed!(2)
    N = 5
    
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
    
    H = heisenberg_1D(N, -1.0, -1.0, -1.0, z=0.1)
    
    println("\nOriginal Hamiltonian: $(length(H)) terms")
    
    # Single run with verbose output
    println("\n" * "="^70)
    println("SINGLE RUN WITH ADAPTIVE TRUNCATION")
    println("="^70)
    O_final, t, hist, metrics = dbf_diag_rk4_fast(H,
                                                   t_end=3.0,
                                                   dt=0.001,
                                                   max_terms=400,
                                                   coeff_thresh=1e-6,
                                                   max_weight=4,
                                                   adaptive_truncation=true,
                                                   verbose=1)
    
    # Compare strategies
    compare_truncation_strategies(H, t_end=2.0, dt=0.01)
# end