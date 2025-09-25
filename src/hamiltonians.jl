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
