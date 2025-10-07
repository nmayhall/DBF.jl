using PauliOperators
using DBF

# Test Hubbard 2D
println("Major-Row Ordering: Hubbard 2D Hamiltonian")
H = DBF.fermi_hubbard_2D(2, 2, 5.0, 2.0)
display(H)
println("Number of terms in Hubbard 2x1 Hamiltonian: ", length(H))
Hmat = Matrix(H)
evals = eigvals(Hmat)
@show minimum(evals)

println("\nColumn-wise (Snake) Ordering: Hubbard 2D Hamiltonian")
H = DBF.fermi_hubbard_2D_snake(2, 2, 5.0, 2.0, snake_ordering=true)
display(H)
println("Number of terms in Hubbard 2x1 Hamiltonian: ", length(H))
Hmat = Matrix(H)
evals = eigvals(Hmat)
@show minimum(evals)
