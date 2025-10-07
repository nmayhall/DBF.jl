using PauliOperators

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



"""
    heisenberg_2D(Nx, Ny, Jx, Jy, Jz; x=0, y=0, z=0, periodic=true)

Create a 2D Heisenberg Hamiltonian on an Nx × Ny square lattice.

# Arguments
- `Nx`, `Ny`: Lattice dimensions  
- `Jx`, `Jy`, `Jz`: Nearest-neighbor coupling constants
- `x`, `y`, `z`: External magnetic field components
- `periodic`: Whether to use periodic boundary conditions (default: true)

# Returns
- `PauliSum` representing the 2D Heisenberg Hamiltonian

# Notes
The Hamiltonian has the form:
H = Σ_{⟨i,j⟩} [-2Jx σᵢˣσⱼˣ - 2Jy σᵢʸσⱼʸ - 2Jz σᵢᶻσⱼᶻ] + Σᵢ [x σᵢˣ + y σᵢʸ + z σᵢᶻ]

Sites are indexed in row-major order: site (i,j) → index = i + j*Nx + 1
"""
function heisenberg_2D(Nx, Ny, Jx, Jy, Jz; x=0, y=0, z=0, periodic=true)
    N_total = Nx * Ny
    H = PauliSum(N_total, Float64)
    
    # Helper function to convert 2D coordinates to 1D index (1-based)
    coord_to_index(i, j) = i + j * Nx + 1
    
    # Nearest-neighbor interactions
    for j in 0:Ny-1  # Row index
        for i in 0:Nx-1  # Column index
            current_site = coord_to_index(i, j)
            
            # Right neighbor (i+1, j)
            if i < Nx - 1 || periodic
                right_i = periodic ? (i + 1) % Nx : i + 1
                right_site = coord_to_index(right_i, j)
                
                H += -2*Jx * Pauli(N_total, X=[current_site, right_site])
                H += -2*Jy * Pauli(N_total, Y=[current_site, right_site])
                H += -2*Jz * Pauli(N_total, Z=[current_site, right_site])
            end
            
            # Up neighbor (i, j+1)  
            if j < Ny - 1 || periodic
                up_j = periodic ? (j + 1) % Ny : j + 1
                up_site = coord_to_index(i, up_j)
                
                H += -2*Jx * Pauli(N_total, X=[current_site, up_site])
                H += -2*Jy * Pauli(N_total, Y=[current_site, up_site])
                H += -2*Jz * Pauli(N_total, Z=[current_site, up_site])
            end
        end
    end
    
    # External magnetic field terms
    for site in 1:N_total
        H += x * Pauli(N_total, X=[site])
        H += y * Pauli(N_total, Y=[site]) 
        H += z * Pauli(N_total, Z=[site])
    end
    
    return H
end

# - - - - - - - - - - - - -
# Fermionic Hamiltonians
# - - - - - - - - - - - - -
"""
   HELPERS
 The following functions are designed to perform the Jordan-Wigner mapping.
"""
# --- helpers for JWmapping ---
@inline ubit(i::Int) = UInt128(1) << (i-1)                         # bit at site i (1-based)
@inline umask_lt(i::Int) = i==1 ? UInt128(0) : (ubit(i) - UInt128(1))  # bits < i
@inline umask_le(i::Int) = umask_lt(i) | ubit(i)                      # bits ≤ i

# --- JW mapping (original real ± form; your coeff() supplies +i on ZX sites) ---
function JWmapping(o::Pauli{N}; i::Int, j::Int) where N
    1 <= i <= N || throw(DimensionMismatch("site i=$i out of 1:$N"))
    1 <= j <= N || throw(DimensionMismatch("site j=$j out of 1:$N"))

    # X pieces with Z-strings
    ax = Pauli{N}(1, reinterpret(Int128, umask_lt(i)), reinterpret(Int128, ubit(i)))  # Z^{<i} X_i
    bx = Pauli{N}(1, reinterpret(Int128, umask_lt(j)), reinterpret(Int128, ubit(j)))  # Z^{<j} X_j

    # "Y" pieces = Z^{≤i} X_i, Z^{≤j} X_j  (no explicit im; your coeff() turns ZX into iY)
    ay = Pauli{N}(1, reinterpret(Int128, umask_le(i)), reinterpret(Int128, ubit(i)))
    by = Pauli{N}(1, reinterpret(Int128, umask_le(j)), reinterpret(Int128, ubit(j)))

    # c†_i = (X_i - Y_i)/2,  c_j = (X_j + Y_j)/2   in your convention
    c_dagg_i = 0.5 * (ax - ay)
    c_j      = 0.5 * (bx + by)

    return c_dagg_i * c_j
end

"""
 1D Fermi-Hubbard model 
Generate a 1D Fermi-Hubbard Hamiltonian (open boundaries, no PBC)
using JW mapping into Pauli operators.

Arguments:
- o::Pauli{N} : reference Pauli object
- L::Int       : number of sites
- t::Float64   : hopping amplitude
- U::Float64   : on-site interaction
- k::Int       : number of Trotter steps (can be used later for evolution)

Returns:
- generators::Vector{Pauli{N}}
- parameters::Vector{Float64}
"""

function hubbard_model_1D(L::Int64, t::Float64, U::Float64)
    
    N_total = 2 * L   # Total number of fermionic modes (spin up and down)
    H = PauliSum(N_total, Float64)

    # Hopping terms
    for i in 1:L-1
        # spin-up
        a_up = 2*i - 1
        b_up = 2*(i+1) - 1
        hopping_up = JWmapping(N_total, i=a_up, j=b_up) + JWmapping(N_total, i=b_up, j=a_up)

        # spin-down
        a_dn = 2*i
        b_dn = 2*(i+1)
        hopping_dn = JWmapping(N_total, i=a_dn, j=b_dn) + JWmapping(N_total, i=b_dn, j=a_dn)
        
        # Add both
        H += -t * (hopping_up + hopping_dn)
    end

    # On-site interaction terms``
    for i in 1:L
        a_up = 2*i - 1   # spin-up orbital index
        a_dn = 2*i       # spin-down orbital index
        interaction_term = U *JWmapping(N_total, i=a_up, j=a_up) * JWmapping(N_total, i=a_dn, j=a_dn)

        H += interaction_term
    end

    #Filter zero coefficients
    DBF.coeff_clip!(H)

    return H    
end


"""
    fermi_hubbard_2D(Lx, Ly, t, U; reverse_ordering=false)

Construct generators and parameters for the 2D spinful Hubbard model on Lx×Ly
(physical sites). Each physical site has two spin-orbitals (up, down), so
total qubits N must equal 2 * Lx * Ly.

Returns (generators::Vector{Pauli{N}}, parameters::Vector{Float64}).
"""
function fermi_hubbard_2D(Lx::Int, Ly::Int, t::Float64, U::Float64)
    Nsites = Lx * Ly
    N_total = 2 * Nsites   # Total number of fermionic modes (spin up and down)
    H = PauliSum(N_total, Float64)

    if 2 * Nsites != N_total
        throw(ArgumentError("Total qubits N must equal 2 * Lx * Ly. Got N=$N_total, Lx*Ly=$Nsites"))
    end

    up(j) = 2*j - 1
    dn(j) = 2*j
    linear_index(x,y) = (x - 1) * Ly + y   # x in 1:Lx, y in 1:Ly

    # small tolerance for dropping tiny coeffs
    eps_coeff = 1e-12

    # HOPPING: loop nearest-neighbour pairs once, add c_i^† c_j + c_j^† c_i (both spins)
    for x in 1:Lx, y in 1:Ly
        jsite = linear_index(x, y)
         # neighbor +x (right in x)
        if x < Lx
            isite = linear_index(x + 1, y)
            for spin in (up, dn)
                m = spin(jsite)   # mode index for j
                n = spin(isite)   # mode index for i
                term = JWmapping(N_total, i=m, j=n) + JWmapping(N_total, i=n, j=m)
                H += -t * term
            end
        end
         # neighbor +y (right in y)
         if y < Ly
            isite = linear_index(x, y + 1)
            for spin in (up, dn)
                m = spin(jsite)
                n = spin(isite)
                term = JWmapping(N_total, i=m, j=n) + JWmapping(N_total, i=n, j=m)
                H += -t * term
            end
        end
    end

    for i in 1:Nsites
        a_up = 2*i - 1   # spin-up orbital index
        a_dn = 2*i       # spin-down orbital index
        interaction_term = U *JWmapping(N_total, i=a_up, j=a_up) * JWmapping(N_total, i=a_dn, j=a_dn)

        H += interaction_term
    end

    # Filter zero coefficients
    DBF.coeff_clip!(H, thresh=eps_coeff)

    return H
end


# Test Hubbard 1D
H = hubbard_model_1D(2, 5.0, 2.0)
display(H)
println("Number of terms in Hubbard 1D Hamiltonian: ", length(H))

# Test Hubbard 2D
H = fermi_hubbard_2D(1, 2, 5.0, 2.0)
display(H)
println("Number of terms in Hubbard 2x1 Hamiltonian: ", length(H))