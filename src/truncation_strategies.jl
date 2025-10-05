using PauliOperators
using LinearAlgebra
using Printf

"""
Truncation strategies for efficient DBF dynamics.

Key insight: Truncate *during* operations, not just after.
This prevents intermediate explosion of terms.
"""


"""
    truncate_topk!(O::PauliSum{N}, k::Int) where N

Keep only the k terms with largest absolute coefficients.
This preserves the most "important" terms energetically.
"""
function truncate_topk!(O::PauliSum{N}, k::Int) where N
    if length(O) <= k
        return O
    end
    
    # Get top k by absolute value
    top_terms = partialsort(collect(O), 1:k, by=p->abs(p.second), rev=true)
    
    # Clear and rebuild
    empty!(O)
    for (p, c) in top_terms
        O[p] = c
    end
    
    return O
end


"""
    truncate_combined!(O::PauliSum{N}; 
                       max_terms=1000,
                       coeff_thresh=1e-8,
                       max_weight=nothing) where N

Combined truncation: Apply multiple criteria in order of speed.
- First: Remove tiny coefficients (fast)
- Second: Weight clipping (medium)  
- Third: Top-k selection if still too large (slow but effective)
"""
function truncate_combined!(O::PauliSum{N}; 
                           max_terms=1000,
                           coeff_thresh=1e-8,
                           max_weight=nothing) where N
    
    # Stage 1: Coefficient threshold (very fast)
    if coeff_thresh !== nothing
        filter!(p -> abs(p.second) > coeff_thresh, O)
    end
    
    # Stage 2: Weight clipping (fast)
    if max_weight !== nothing
        filter!(p -> weight(p.first) <= max_weight, O)
    end
    
    # Stage 3: Top-k if still too large (slower but effective)
    if max_terms !== nothing && length(O) > max_terms
        truncate_topk!(O, max_terms)
    end
    
    return O
end


"""
    smart_commutator(A::PauliSum{N}, B::PauliSum{N}; 
                    max_terms=1000,
                    coeff_thresh=1e-8) where N

Compute [A, B] with intelligent intermediate truncation.

Strategy:
1. Pre-truncate A and B to most important terms
2. Only compute commutators for term pairs likely to matter
3. Accumulate result with running truncation
"""
function smart_commutator(A::PauliSum{N}, B::PauliSum{N}; 
                         max_terms=1000,
                         coeff_thresh=1e-8,
                         n_top=nothing) where N
    
    # Determine how many terms to keep from each
    if n_top === nothing
        n_top = min(length(A), length(B), max_terms)
    end
   
    # return A*B-B*A
    # Pre-truncate to top terms
    A_terms = length(A) > n_top ? partialsort(collect(A), 1:n_top, by=p->abs(p.second), rev=true) : collect(A)
    B_terms = length(B) > n_top ? partialsort(collect(B), 1:n_top, by=p->abs(p.second), rev=true) : collect(B)
    
    result = PauliSum(N)
    sizehint!(result, min(max_terms, n_top * n_top))
    
    # Compute commutator with early truncation
    count = 0
    for (p1, c1) in A_terms
        for (p2, c2) in B_terms
            # Only compute if terms don't commute
            if !PauliOperators.commute(p1, p2)
                prod = 2 * c1 * c2 * (p1 * p2)
                curr = get(result, PauliBasis(prod), 0.0) + PauliOperators.coeff(prod)
                result[PauliBasis(prod)] = curr
                
                count += 1
                # # Periodically truncate during accumulation
                # if count % 1000 == 0 && length(result) > max_terms * 2
                #     truncate_combined!(result, max_terms=max_terms, coeff_thresh=coeff_thresh)
                # end
            end
        end
    end
    
    # Final truncation
    truncate_combined!(result, max_terms=max_terms, coeff_thresh=coeff_thresh)
    
    return result
end


"""
    smart_double_commutator(O::PauliSum{N};
                           max_terms=1000,
                           coeff_thresh=1e-8,
                           max_weight=nothing) where N

Compute [O, [O, diag(O)]] with aggressive truncation at each stage.

This is much faster than the naive double commutator.
"""
function smart_double_commutator(O::PauliSum{N};
                                max_terms=1000,
                                coeff_thresh=1e-8,
                                max_weight=nothing) where N
    
    # Get diagonal part (always fast)
    Od = diag(O)
    
    # First commutator with truncation
    comm1 = smart_commutator(O, Od, 
                            max_terms=max_terms, 
                            coeff_thresh=coeff_thresh)
    
    # Apply weight clipping to first commutator if requested
    if max_weight !== nothing
        filter!(p -> weight(p.first) <= max_weight, comm1)
    end
    
    # Second commutator with truncation
    comm2 = smart_commutator(O, comm1,
                            max_terms=max_terms,
                            coeff_thresh=coeff_thresh)
    
    # Final cleanup
    truncate_combined!(comm2, 
                      max_terms=max_terms,
                      coeff_thresh=coeff_thresh,
                      max_weight=max_weight)
    
    return comm2
end


"""
    adaptive_truncation_params(O::PauliSum; 
                              base_max_terms=1000,
                              base_coeff_thresh=1e-8)

Adaptively adjust truncation parameters based on current operator size.

If operator is growing too large, become more aggressive.
"""
function adaptive_truncation_params(O::PauliSum; 
                                   base_max_terms=1000,
                                   base_coeff_thresh=1e-8,
                                   target_size=500)
    
    current_size = length(O)
    
    if current_size < target_size
        # Operator is small, be lenient
        return (max_terms = base_max_terms * 2,
                coeff_thresh = base_coeff_thresh / 10)
    elseif current_size < base_max_terms
        # Normal regime
        return (max_terms = base_max_terms,
                coeff_thresh = base_coeff_thresh)
    else
        # Operator is large, be aggressive
        scale = current_size / base_max_terms
        return (max_terms = base_max_terms,
                coeff_thresh = base_coeff_thresh * scale)
    end
end


"""
    locality_preserving_truncate!(O::PauliSum{N}, max_distance::Int) where N

Keep only terms where qubits are within max_distance of each other.

For a 1D chain, this preserves locality. Useful if you want to maintain
a physical constraint that interactions are short-range.
"""
function locality_preserving_truncate!(O::PauliSum{N}, max_distance::Int) where N
    filter!(O) do (p, c)
        bits = sort(collect(PauliOperators.get_on_bits(p.x | p.z)))
        if length(bits) <= 1
            return true
        end
        # Check if all qubits are within max_distance
        return (bits[end] - bits[1]) <= max_distance
    end
    return O
end


"""
    relative_truncate!(O::PauliSum, ratio::Float64)

Keep only terms with |coefficient| > ratio * max(|coefficients|).

This is more adaptive than fixed thresholding.
"""
function relative_truncate!(O::PauliSum, ratio::Float64=0.001)
    if isempty(O)
        return O
    end
    
    max_coeff = maximum(abs(c) for (p, c) in O)
    thresh = ratio * max_coeff
    
    filter!(p -> abs(p.second) > thresh, O)
    return O
end


# Helper function
function weight(p::PauliBasis) 
    return count_ones(p.x | p.z)
end


# Demonstration
if abspath(PROGRAM_FILE) == @__FILE__
    using Random
    using BenchmarkTools
    
    println("=== Testing Truncation Strategies ===\n")
    
    Random.seed!(2)
    N = 6
    
    # Create test operator
    O = rand(PauliSum{N}, n_paulis=500)
    O = O + O'  # Make Hermitian
    
    println("Original operator: $(length(O)) terms\n")
    
    # Test 1: Naive double commutator
    println("--- Naive double commutator ---")
    O1 = deepcopy(O)
    @time begin
        Od = diag(O1)
        comm1 = O1 * Od - Od * O1
        comm2 = O1 * comm1 - comm1 * O1
    end
    println("Result: $(length(comm2)) terms")
    println("Norm: $(norm(comm2))\n")
    
    # Test 2: Smart double commutator
    println("--- Smart double commutator ---")
    @time comm2_smart = smart_double_commutator(O,
                                                max_terms=500,
                                                coeff_thresh=1e-6,
                                                max_weight=4)
    println("Result: $(length(comm2_smart)) terms")
    println("Norm: $(norm(comm2_smart))\n")
    
    # Test 3: Different truncation strategies
    println("--- Truncation strategy comparison ---")
    test_op = rand(PauliSum{N}, n_paulis=2000)
    
    println("Original: $(length(test_op)) terms")
    
    op1 = deepcopy(test_op)
    truncate_topk!(op1, 500)
    println("Top-k (500): $(length(op1)) terms, norm=$(norm(op1))")
    
    op2 = deepcopy(test_op)
    truncate_combined!(op2, max_terms=500, coeff_thresh=1e-4)
    println("Combined: $(length(op2)) terms, norm=$(norm(op2))")
    
    op3 = deepcopy(test_op)
    relative_truncate!(op3, 0.01)
    println("Relative (1%): $(length(op3)) terms, norm=$(norm(op3))")
    
    println("\nOriginal norm: $(norm(test_op))")
end