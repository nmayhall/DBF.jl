using PauliOperators
using LinearAlgebra
using DataStructures

function expectation_value_dfs(
    P::PauliBasis{N},
    generators::Vector{PauliBasis{N}},
    angles::Vector{Float64},
    psi::Ket{N},
    stack_paulis::Vector{PauliBasis{N}},
    stack_coeffs::Vector{ComplexF64},
    stack_depths::Vector{Int}
) where N
    
    n_gen = length(generators)
    
    # Initialize stack with the starting Pauli
    stack_ptr = 1
    stack_paulis[1] = P
    stack_coeffs[1] = ComplexF64(1.0)
    stack_depths[1] = 1
    
    result = ComplexF64(0.0)
    
    while stack_ptr > 0
        # Pop from stack
        current_P = stack_paulis[stack_ptr]
        current_coeff = stack_coeffs[stack_ptr]
        depth = stack_depths[stack_ptr]
        stack_ptr -= 1
        
        # Process remaining generators
        @inbounds for k in depth:n_gen
            G = generators[k]
            θ = angles[k]
            
            if PauliOperators.commute(current_P, G)
                continue
            else
                c = cos(θ)
                s = sin(θ)
                
                # Compute G*P
                GP_prod = Pauli(G) * Pauli(current_P)
                GP_basis = PauliBasis(GP_prod)
                GP_phase = coeff(GP_prod)
                
                # Push sin branch
                sin_coeff = current_coeff * 1im * s * GP_phase
                # if abs2(sin_coeff) > 1e-2
                    stack_ptr += 1
                    stack_paulis[stack_ptr] = GP_basis
                    stack_coeffs[stack_ptr] = sin_coeff 
                    stack_depths[stack_ptr] = k + 1
                # end
                # Continue with cos branch
                current_coeff *= c
                current_P = current_P  # unchanged basis for cos branch
            end
        end
        
        result += current_coeff * expectation_value(current_P, psi)
    end
    
    return result
end

function expectation_value_dfs(
    P::PauliBasis{N},
    generators::Vector{PauliBasis{N}},
    angles::Vector{Float64},
    psi::Ket{N}
) where N
    n_gen = length(generators)
    
    stack_paulis = Vector{PauliBasis{N}}(undef, n_gen + 1)
    stack_coeffs = Vector{ComplexF64}(undef, n_gen + 1)
    stack_depths = Vector{Int}(undef, n_gen + 1)
    
    return expectation_value_dfs(P, generators, angles, psi, 
                                        stack_paulis, stack_coeffs, stack_depths)
end