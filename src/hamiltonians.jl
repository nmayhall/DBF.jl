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