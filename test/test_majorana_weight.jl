using PauliOperators
using LinearAlgebra
using DBF


#Example Types definitions
Pb = PauliBasis("ZIXXXZZZXXY")
println("- - - PauliBasis Properties - - - ")

println("Example, pauli basis: ", Pb)
println("ishermitian: ", LinearAlgebra.ishermitian(Pb))
println("Coefficient: ", coeff(Pb))
println("Symplectic Phase: ", symplectic_phase(Pb))
println("Y-Matrix Representation : ", LinearAlgebra.Matrix(PauliBasis("Y")))

w = DBF.majorana_weight(Pb)
println("Majorana weight is: ", w)

# Test weight functions in helpers
function test_weights()
    N = 4
    n_lower = 0

    #Example of Paulis
    all_paulis = ["IIII", "IIIX", "IIXI", "IIXX", "IXII", "IXIX", "IXXI", "IXXX",
                  "XIII", "XIIX", "XIXI", "XIXX", "XXII", "XXIX", "XXXI", "XXXX",
                  "ZZZZ", "ZZZY", "ZZYZ", "ZZYY", "ZYZZ", "ZYZY", "ZYYZ", "ZYYY",
                  "YZZZ", "YZZY", "YZYZ", "YZYY", "YYZZ", "YYZY", "YYYZ", "YYYY",
                  "XXZZ", "XXZY", "XXYZ", "XXYY", "XZXX", "XZXZ", "XZZX", "XZZZ",
                  "YXXZ", "YXXY", "YXZX", "YXZY", "YYXZ", "YYXY", "YYZX", "YYZY"]
    total_paulis = length(all_paulis)


    println("Test weights")
    for (i,op) in enumerate(all_paulis)
        # Extract the Pauli string from our custom example 
        P = PauliBasis(op)

        pauli_weight = DBF.pauli_weight(P)
        majo_weight = DBF.majorana_weight(P)

        println("$(i): $op Pauli weight $pauli_weight ==> Majorana weight $majo_weight")
    end

    println("Majorana monomials with weight lower than Paulis: $n_lower from $total_paulis total operators")


    # # Show example cases
    println("\n- - - Example Cases - - - ")
    println("Example case: YXXZZZ")
    pauli_basis = PauliBasis("YXXZZZ")
    println("Majorana weight:")
    println(DBF.majorana_weight(pauli_basis))
    println("Pauli weight:")
    println(DBF.pauli_weight(pauli_basis))

    println("Example case: ZZ")
    pauli_basis = PauliBasis("ZZ")
    println("Majorana weight:")
    println(DBF.majorana_weight(pauli_basis))
    println("Pauli weight:")
    println(DBF.pauli_weight(pauli_basis))

    println("Example case: IXXI")
    pauli_basis = PauliBasis("IXXI")
    println("Majorana weight:")
    println(DBF.majorana_weight(pauli_basis))
    println("Pauli weight:")
    println(DBF.pauli_weight(pauli_basis))

    println("Example case: YZZY")
    pauli_basis = PauliBasis("YZZY")
    println("Majorana weight:")
    println(DBF.majorana_weight(pauli_basis))
    println("Pauli weight:")
    println(DBF.pauli_weight(pauli_basis))
end


# Run tester if this file is executed directly (script mode)
test_weights()