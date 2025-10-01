using Test
using Random
using FFTW
using Plots
using Printf

# @testset "theta_opt" begin
function test()
    N = 2
    H = DBF.heisenberg_1D(N,1,1,1,z=.1)
    O = PauliSum(Pauli(N,Z=[1]))
    O = rand(PauliSum{N}); O+=O'; O = O*(1/norm(O))
    ψ = Ket{N}(0)
    display(O)
   
    display(eigvals(Matrix(H)))
    G = Vector{PauliBasis{N}}([])
    h = Vector{Float64}([])
    for (p,c) in H
        push!(G, p)
        push!(h, c)
    end

    dt = .01
    nsteps = 10000
    signal_Ot = Vector{Float64}([])
    signal_O0Ot = Vector{Float64}([])
    O0 = deepcopy(O)
    Ot = expectation_value(O,ψ)
    O0Ot = inner_product(O0,O) 
    # @printf(" %12s %12s %12s %8s %12s\n", "t", "<O(t)>", "<O(0)O(t)>", "len(O)", "norm(O)")

    for ti in 0:nsteps-1
        for (gi,hi) in zip(G,h)
            O = evolve(O,gi,hi*dt)
            # DBF.coeff_clip!(O,thresh=1e-4)
        end
        # DBF.dissipate!(O,2,.1*dt)
        Ot = expectation_value(O,ψ)
        O0Ot = inner_product(O0,O) 
        
        push!(signal_Ot, Ot)
        push!(signal_O0Ot, O0Ot)

        # @printf(" %12.8f %12.8f %12.8f %8i %12.8f\n", ti*dt, Ot, O0Ot, length(O), norm(O))
    end
    dt = .002
    f = 16
    signal_O0Ot = [exp(1im*f*ti*dt) for ti in 0:nsteps-1 ]
    # signal_O0Ot = [exp(-1im*(f-.0im)*ti*dt) for ti in 0:nsteps-1 ]
    plot_signal_and_dft(signal_O0Ot, 1/dt)
    freqs, rates, amplitudes = filter_diagonalization_method(signal_O0Ot, 1/dt)
   
    @show length(freqs), length(rates), length(amplitudes)
    for (f,r,a) in zip(freqs,rates,amplitudes)
        @printf(" %12.8f %12.8f %12.8f+%12.8fi\n", f, r, real(a), imag(a))
    end

end

function compute_dft(signal)
    """
    Compute the Discrete Fourier Transform of a signal.
    
    Args:
        signal: Input signal (vector of numbers)
    
    Returns:
        DFT coefficients (complex numbers)
    """
    return fft(signal)
end

function plot_signal_and_dft(signal, fs)
    """
    Plot the original signal and its DFT magnitude spectrum.
    
    Args:
        signal: Input signal
        fs: Sampling frequency (default: 1.0 Hz)
    """
    N = length(signal)
    
    # Compute DFT
    dft_result = compute_dft(signal)
    
    # Create time and frequency vectors
    t = (0:N-1) / fs
    freqs = (0:N-1) * fs / N * 2π 
    
    # Only plot positive frequencies (first half)
    # half_N = div(N, 2) + 1
    # freqs_positive = freqs[1:half_N]
    magnitude = abs.(dft_result)
    # magnitude = abs.(dft_result[1:half_N])
    
    # Create plots
    p1 = plot(t, real.(signal), 
              title="Original Signal", 
              xlabel="Time (s)", 
              ylabel="Amplitude",
              linewidth=2,
              color=:blue)
    p1 = plot!(t, imag.(signal), 
              title="Original Signal", 
              xlabel="Time (s)", 
              ylabel="imag(Amplitude)",
              linewidth=1,
              color=:green)
    
    p2 = plot(freqs, magnitude, 
              title="DFT Magnitude Spectrum", 
              xlabel="Frequency (Hz)", 
              ylabel="Magnitude",
              xlims=(-20,20),
              linewidth=2,
              color=:red)
    
    # Combine plots
    p = plot(p1, p2, layout=(2,1), size=(600, 500))
    savefig(p, "freq.pdf")

end

function method2_prony_method(signal, fs, num_modes=6)
    """
    Method 2: Prony's method for parametric estimation
    Better for short signals with exponential decay
    """
    N = length(signal)
    
    # Create prediction matrix for linear prediction
    # This is a simplified version of Prony's method
    
    # Form the data matrix
    p = min(num_modes * 2, N ÷ 3)  # Prediction order
    
    if N <= p + 1
        return [], []  # Signal too short
    end
    
    # Create Hankel matrix
    A = zeros(N - p, p)
    for i in 1:(N-p)
        A[i, :] = signal[i:i+p-1]
    end
    
    b = signal[p+1:N]
    
    # Solve linear prediction coefficients
    coeffs = A \ b
    try
        coeffs = A \ b
    catch
        return [], []  # Singular matrix
    end
    
    # Form characteristic polynomial
    poly_coeffs = [1; -coeffs]
    
    # Find roots (poles)
    roots_poly = roots(poly_coeffs)
    
    # Convert to frequencies and damping
    dt = 1/fs
    frequencies = []
    dampings = []
    
    for root in roots_poly
        if abs(imag(root)) > 1e-10  # Complex roots only
            # Convert pole to frequency and damping
            magnitude = abs(root)
            phase = angle(root)
            
            if magnitude < 1.0  # Stable pole
                freq = abs(phase) / (2π * dt)
                damping = -log(magnitude) / dt
                
                if freq > 0 && freq < fs/2  # Valid frequency range
                    push!(frequencies, freq)
                    push!(dampings, damping)
                end
            end
        end
    end
    
    return frequencies, dampings
end


using LinearAlgebra
using FFTW
using Plots
function filter_diagonalization_method(signal, fs; M=nothing, filter_type=:exponential)
    """
    Filter Diagonalization Method (FDM) for harmonic inversion
    
    This method extracts frequencies and decay rates from short time signals
    by constructing and diagonalizing a filter matrix.
    
    Args:
        signal: Time domain signal
        fs: Sampling frequency
        M: Filter matrix size (default: N/3)
        filter_type: Type of filter (:exponential, :gaussian, :lorentzian)
    
    Returns:
        frequencies: Extracted frequencies (Hz)
        decay_rates: Decay rates (1/s)
        amplitudes: Complex amplitudes
    """
    
    N = length(signal)
    if M === nothing
        M = min(N ÷ 3, 50)  # Typical choice: N/3 but not too large
    end
    
    # Ensure M is reasonable
    M = max(2, min(M, N ÷ 2 - 1))  # More conservative bound
    
    # Additional safety check
    if M >= N - 1
        @warn "Signal too short for FDM analysis. Need at least $(2*M + 2) samples, got $N"
        return Float64[], Float64[], ComplexF64[]
    end
    
    dt = 1.0 / fs
    
    # Step 1: Create the signal correlation matrix (Hankel matrix)
    # This is similar to what we did in Prony's method but more sophisticated
    U = zeros(ComplexF64, M, N - M + 1)
    
    for j in 1:(N - M + 1)
        U[:, j] = signal[j:j + M - 1]
    end
    
    # Step 2: Choose and apply filter function
    # The filter function determines the resolution and stability
    num_lags = N - M + 1
    filter_weights = create_filter_weights(num_lags, filter_type)
    
    # Apply filter to create filtered correlation matrix
    F = zeros(ComplexF64, M, M)
    
    for i in 1:M
        for j in 1:M
            # This is the key FDM step: filtered inner product
            for k in 1:num_lags
                idx_i = i + k - 1
                idx_j = j + k - 1
                if idx_i <= N && idx_j <= N
                    F[i, j] += conj(signal[idx_i]) * signal[idx_j] * filter_weights[k]
                end
            end
        end
    end
    
    # Step 3: Create the generalized eigenvalue problem
    # Construct B matrix (shifted version of F)
    B = zeros(ComplexF64, M, M)
    
    for i in 1:(M-1)
        for j in 1:(M-1)
            for k in 1:(num_lags-1)
                idx_i = i + k
                idx_j = j + k + 1
                if idx_i <= N && idx_j <= N && k <= length(filter_weights)
                    B[i, j] += conj(signal[idx_i]) * signal[idx_j] * filter_weights[k]
                end
            end
        end
    end
    
    # Step 4: Solve generalized eigenvalue problem F * v = λ * B * v
    # The eigenvalues λ contain the frequency and decay information
    
    # Check for numerical issues
    if any(isnan.(F)) || any(isnan.(B))
        @warn "NaN values detected in matrices"
        return Float64[], Float64[], ComplexF64[]
    end
   
    eigenvals = Float64[]
    eigenvecs = Float64[]

    try
        eigenvals, eigenvecs = eigen(B, F)
    catch e
        # Fallback to regular eigenvalue problem if generalized fails
        @warn "Generalized eigenvalue problem failed, trying fallback: $e"
        try
            if rank(F) < size(F, 1)
                @warn "F matrix is singular, using pseudoinverse"
            end
            F_inv = pinv(F)
            eigenvals, eigenvecs = eigen(F_inv * B)
        catch e2
            @warn "Both eigenvalue methods failed: $e2"
            return Float64[], Float64[], ComplexF64[]
        end
    end
    
    # Step 5: Extract frequencies and decay rates from eigenvalues
    frequencies = Float64[]
    decay_rates = Float64[]
    amplitudes = ComplexF64[]
    
    for (i, λ) in enumerate(eigenvals)
        if abs(λ) > 1e-12  # Avoid numerical zeros
            # Convert eigenvalue to complex frequency
            s = log(λ) / dt
            
            # Extract frequency and decay rate
            frequency = abs(imag(s)) / (2π)
            decay_rate = -real(s)
            
            # Filter reasonable results
            if (0 < frequency < fs/2 && 
                decay_rate > 0 && 
                decay_rate < 1000)  # Reasonable bounds
                
                push!(frequencies, frequency)
                push!(decay_rates, decay_rate)
                
                # Estimate amplitude (simplified)
                push!(amplitudes, eigenvecs[1, i])
            end
        end
    end
    
    # Sort by frequency
    if !isempty(frequencies)
        sort_idx = sortperm(frequencies)
        frequencies = frequencies[sort_idx]
        decay_rates = decay_rates[sort_idx]
        amplitudes = amplitudes[sort_idx]
    end
    
    return frequencies, decay_rates, amplitudes
end

function create_filter_weights(num_points, filter_type)
    """
    Create filter weights for different filter types
    """
    k = 0:(num_points-1)
    
    if filter_type == :exponential
        # Exponential filter (most common)
        α = 0.1  # Filter parameter
        return exp.(-α * k)
        
    elseif filter_type == :gaussian
        # Gaussian filter
        σ = num_points / 4
        return exp.(-k.^2 / (2σ^2))
        
    elseif filter_type == :lorentzian
        # Lorentzian filter
        γ = num_points / 10
        return 1 ./ (1 .+ (k / γ).^2)
        
    else
        # Uniform filter (equivalent to no filter)
        return ones(num_points)
    end
end


test()