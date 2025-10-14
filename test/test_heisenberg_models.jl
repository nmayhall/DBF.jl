using PauliOperators
using DBF

Lx=10
Ly=10
println()
periodic=false #OCT-pending to revise and correct error with periodic boundary conditions
snake_ordering=true

H = DBF.heisenberg_2D_snake(Lx, Ly, -1.0,-1.0,-1.0, periodic=periodic, snake_ordering=snake_ordering)
display(H)

println("Number of terms in Heisenberg 2D Hamiltonian: ", length(H))
pauli_strings=[]
coeffs=[]
for (c,p) in H
   println(string(c))
   push!(pauli_strings,string(c))
   push!(coeffs,real(p))
end
display(coeffs)
display(pauli_strings)
