using PauliOperators

"""
    rk4_step(f, t, y, dt)

Perform a single RK4 step for the differential equation dy/dt = f(t, y).

# Arguments
- `f`: Function that computes the right-hand side, signature `f(t, y) -> PauliSum`
- `t`: Current time
- `y`: Current state (PauliSum)
- `dt`: Time step

# Returns
- `y_new`: Updated state at time t + dt
"""
function rk4_step(f, t, y, dt)
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    
    y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
   
    # @show length(y)
    return y_new
end


"""
    rk4_integrate(f, y0, tspan, dt; 
                  callback=nothing, 
                  truncate_coeff=nothing,
                  truncate_weight=nothing)

Integrate dy/dt = f(t, y) using RK4 method.

# Arguments
- `f`: Right-hand side function, signature `f(t, y) -> PauliSum`
- `y0`: Initial condition (PauliSum)
- `tspan`: Tuple `(t_start, t_end)` defining integration interval
- `dt`: Time step

# Keyword Arguments
- `callback`: Optional function called after each step as `callback(t, y)`. 
             If it returns `true`, integration stops early.
- `truncate_coeff`: If provided, coefficients smaller than this are removed
- `truncate_weight`: If provided, terms with weight > this are removed

# Returns
- `t_history`: Vector of time points
- `y_history`: Vector of states (PauliSum objects)

# Example
```julia
# Linear oscillator: dy/dt = -i * H * y
H = PauliSum(Pauli(2, X=[1]))
y0 = PauliSum(Pauli(2, Z=[1]))
rhs(t, y) = -1im * H * y

t_hist, y_hist = rk4_integrate(rhs, y0, (0.0, 10.0), 0.01)
```
"""
function rk4_integrate(f, y0, tspan, dt; 
                       callback=nothing,
                       truncate_coeff=nothing,
                       truncate_weight=nothing)
    
    t_start, t_end = tspan
    n_steps = Int(ceil((t_end - t_start) / dt))
   
    
    # Storage
    t_history = Vector{Float64}()
    y_history = Vector{typeof(y0)}()
    
    # Initial condition
    t = t_start
    y = deepcopy(y0)
    
    push!(t_history, t)
    push!(y_history, deepcopy(y))
    
    # Main integration loop
    for step in 1:n_steps
        # Take RK4 step
        y = rk4_step(f, t, y, dt)
        t += dt
        
        # Optional truncation to control growth
        if truncate_coeff !== nothing
            coeff_clip!(y, thresh=truncate_coeff)
        end
        
        if truncate_weight !== nothing
            weight_clip!(y, truncate_weight)
        end
        
        # Store results
        push!(t_history, t)
        push!(y_history, deepcopy(y))
        
        # Optional callback
        if callback !== nothing
            if callback(t, y)
                break  # Stop integration if callback returns true
            end
        end
    end
    
    return t_history, y_history
end


"""
    rk4_integrate_adaptive(f, y0, tspan, dt_init; 
                          tol=1e-6,
                          dt_min=1e-8,
                          dt_max=1.0,
                          callback=nothing,
                          truncate_coeff=nothing,
                          truncate_weight=nothing)

Integrate with adaptive time stepping based on local error estimation.

Uses step doubling: compares one step of size dt with two steps of size dt/2.

# Arguments
- `f`: Right-hand side function
- `y0`: Initial condition (PauliSum)
- `tspan`: Tuple `(t_start, t_end)`
- `dt_init`: Initial time step

# Keyword Arguments
- `tol`: Error tolerance for adaptive stepping
- `dt_min`: Minimum allowed time step
- `dt_max`: Maximum allowed time step
- `callback`: Optional callback function
- `truncate_coeff`: Coefficient truncation threshold
- `truncate_weight`: Weight truncation threshold

# Returns
- `t_history`: Vector of time points
- `y_history`: Vector of states
"""
function rk4_integrate_adaptive(f, y0, tspan, dt_init;
                               tol=1e-6,
                               dt_min=1e-8,
                               dt_max=1.0,
                               callback=nothing,
                               truncate_coeff=nothing,
                               truncate_weight=nothing)
    
    t_start, t_end = tspan
    
    # Storage
    t_history = Vector{Float64}()
    y_history = Vector{typeof(y0)}()
    
    # Initial condition
    t = t_start
    y = deepcopy(y0)
    dt = dt_init
    
    push!(t_history, t)
    push!(y_history, deepcopy(y))
    
    while t < t_end
        # Don't overshoot the end
        if t + dt > t_end
            dt = t_end - t
        end
        
        # One big step
        y1 = rk4_step(f, t, y, dt)
        
        # Two small steps
        y_half = rk4_step(f, t, y, dt/2)
        y2 = rk4_step(f, t + dt/2, y_half, dt/2)
        
        # Estimate error (difference between methods)
        error = norm(y2 - y1)
        
        # Accept or reject step
        if error < tol || dt <= dt_min
            # Accept step
            y = y2  # Use more accurate result
            t += dt
            
            # Truncation
            if truncate_coeff !== nothing
                coeff_clip!(y, thresh=truncate_coeff)
            end
            if truncate_weight !== nothing
                weight_clip!(y, truncate_weight)
            end
            
            # Store
            push!(t_history, t)
            push!(y_history, deepcopy(y))
            
            # Callback
            if callback !== nothing
                if callback(t, y)
                    break
                end
            end
        end
        
        # Adjust step size
        if error > 0
            dt = dt * min(2.0, max(0.5, 0.9 * (tol / error)^0.25))
        else
            dt = dt * 2.0  # Error is zero, try larger step
        end
        
        dt = clamp(dt, dt_min, dt_max)
    end
    
    return t_history, y_history
end