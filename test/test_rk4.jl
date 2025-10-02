using PauliOperators
using LinearAlgebra
using Printf

include("../src/rk4.jl")

"""
Example 1: Simple unitary evolution
Solve: dy/dt = -i * H * y
"""
function example_unitary_evolution()
    println("\n=== Example 1: Unitary Evolution ===")
    
    N = 3
    H = heisenberg_1D(N, 1.0, 1.0, 1.0)
    y0 = PauliSum(Pauli(N, Z=[1]))
    
    # Define right-hand side: dy/dt = -i H y
    rhs(t, y) = -1im * H * y
    
    # Integrate
    t_hist, y_hist = rk4_integrate(rhs, y0, (0.0, 2.0), 0.01)
    
    # Check norm conservation (should be constant)
    println("Time | Norm | Length")
    for i in 1:20:length(t_hist)
        @printf("%6.2f | %8.5f | %6i\n", 
                t_hist[i], norm(y_hist[i]), length(y_hist[i]))
    end
end


"""
Example 2: With truncation
"""
function example_with_truncation(;N = 6)
    println("\n=== Example 2: With Truncation ===")
    
    H = heisenberg_1D(N, 1.0, 1.0, 1.0)
    y0 = PauliSum(Pauli(N, Z=[1]))
    
    rhs(t, y) = -1im * H * y
    
    # Without truncation
    println("\nWithout truncation:")
    t1, y1 = rk4_integrate(rhs, y0, (0.0, 5.0), 0.01)
    @printf("Final length: %i\n", length(y1[end]))
    
    # With truncation
    println("\nWith truncation:")
    t2, y2 = rk4_integrate(rhs, y0, (0.0, 5.0), 0.01,
                          truncate_coeff=1e-6,
                          truncate_weight=3)
    @printf("Final length: %i\n", length(y2[end]))
    
    # Compare norms
    @printf("Norm difference: %.6e\n", abs(norm(y1[end]) - norm(y2[end])))
end


"""
Example 3: With callback
"""
function example_with_callback()
    println("\n=== Example 3: With Callback ===")
    
    N = 3
    H = heisenberg_1D(N, 1.0, 1.0, 1.0)
    y0 = PauliSum(Pauli(N, Z=[1]))
    
    rhs(t, y) = -1im * H * y
    
    # Use closure to track step count
    step_count = Ref(0)
    
    # Stop when operator gets too large
    function my_callback(t, y)
        step_count[] += 1
        n = length(y)
        
        if n > 100
            @printf("Stopping at t=%.2f: length = %i\n", t, n)
            return true  # Stop integration
        end
        
        if mod(step_count[], 50) == 0
            @printf("t=%.2f: length = %i\n", t, n)
        end
        return false  # Continue
    end
    
    t_hist, y_hist = rk4_integrate(rhs, y0, (0.0, 10.0), 0.01,
                                   callback=my_callback)
    
    @printf("Integration stopped at t=%.2f\n", t_hist[end])
end


"""
Example 4: Adaptive time stepping
"""
function example_adaptive()
    println("\n=== Example 4: Adaptive Time Stepping ===")
    
    N = 3
    H = heisenberg_1D(N, 1.0, 1.0, 1.0)
    y0 = PauliSum(Pauli(N, Z=[1]))
    
    rhs(t, y) = -1im * H * y
    
    # Fixed step
    @time t1, y1 = rk4_integrate(rhs, y0, (0.0, 5.0), 0.001)
    @printf("Fixed step: %i evaluations\n", length(t1))
    
    # Adaptive step
    @time t2, y2 = rk4_integrate_adaptive(rhs, y0, (0.0, 5.0), 0.1,
                                         tol=1e-4)
    @printf("Adaptive step: %i evaluations\n", length(t2))
    
    @printf("Norm difference: %.6e\n", abs(norm(y1[end]) - norm(y2[end])))
end


"""
Example 5: Non-Hamiltonian evolution
Custom dynamics with dissipation or other effects
"""
function example_custom_dynamics()
    println("\n=== Example 5: Custom Dynamics ===")
    
    N = 3
    H = heisenberg_1D(N, 1.0, 1.0, 1.0)
    y0 = PauliSum(Pauli(N, Z=[1]))
    
    # Hamiltonian evolution + decay of high-weight terms
    function rhs(t, y)
        hamiltonian_part = -1im * H * y
        
        # Add dissipation: decay high-weight terms
        dissipation = deepcopy(y)
        for (p, c) in dissipation
            w = weight(p)
            if w > 2
                dissipation[p] = -0.1 * c  # Decay rate
            else
                dissipation[p] = 0.0
            end
        end
        
        return hamiltonian_part + dissipation
    end
    
    t_hist, y_hist = rk4_integrate(rhs, y0, (0.0, 5.0), 0.01)
    
    println("Time | Norm | Length | High-weight %")
    for i in 1:50:length(t_hist)
        y = y_hist[i]
        high_weight = sum(abs(c)^2 for (p, c) in y if weight(p) > 2)
        total = sum(abs(c)^2 for (p, c) in y)
        @printf("%6.2f | %8.5f | %6i | %6.1f%%\n", 
                t_hist[i], norm(y), length(y), 100*high_weight/total)
    end
end


# Helper function from the package
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

function weight(p::PauliBasis) 
    return count_ones(p.x | p.z)
end

function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    filter!(p->abs(p.second) > thresh, ps)
end

function weight_clip!(ps::PauliSum{N}, max_weight::Int) where {N}
    filter!(p->weight(p.first) <= max_weight, ps)
end


# Run examples
example_unitary_evolution()
example_with_truncation()
example_with_callback()
example_adaptive()
example_custom_dynamics()