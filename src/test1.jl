using PauliOperators
using DBF
using LinearAlgebra
using Printf
using Random

function test1()
    # Random.seed!(1)
    N = 3
    O = rand(PauliSum{N}, n_paulis=100)
    O += O'
    G = rand(PauliBasis{N})
    ψ = Ket{N}(0)

    display(O)
    display(G)

    Gmat = Matrix(G)
    Omat = Matrix(O)
    ψvec = Vector(ψ)

    function TEO(θ)
        return exp(-1im * θ/2 * Gmat)
    end


    Od = diag(O)
    GOGd = diag(G*O*G)
    comd = diag(G*O-O*G)
    A = overlap(Od,Od) 
    B = overlap(GOGd,GOGd) 
    C = overlap(comd,comd) 
    D = overlap(Od,GOGd) 
    E = overlap(Od,comd) 
    F = overlap(GOGd,comd) 

    @show A, B, C, D, E
    function cost_diag(θ)
        # C(x) = ||diag(U(x)' O U(x))||
        #      = ||diag((cos(x/2)+isin(x/2)G) O (cos(x/2)-isin(x/2)G)) ||
        #      = ||cos(x/2)^2diag(O) + sin(x/2)^2 diag(GOG) + isin(x/2)cos(x/2)diag([G,O]))||
        #
        #      = tr(cos(x/2)^2diag(O)' + sin(x/2)^2 diag(GOG)' - isin(x/2)cos(x/2)diag([G,O])')
        #          *cos(x/2)^2diag(O) + sin(x/2)^2 diag(GOG) + isin(x/2)cos(x/2)diag([G,O])) )
        #
        #      = cos(x/2)^4 (Od|Od) + sin(x/2)^4 (GOGd|GOGd) + sin(x/2)^2cos(x/2)^2 ([G,O]d|[G,O]d)
        #      + 2 cos(x/2)^2 sin(x/2)^2 (Od|GOGd)
        #      + i cos(x/2)^3 sin(x/2)^1 (Od|[G,O]d) 
        #      - i cos(x/2)^3 sin(x/2)^1 ([G,O]d|Od)    
        #      + i cos(x/2)^1 sin(x/2)^3 (GOGd|[G,O]d) 
        #      - i cos(x/2)^1 sin(x/2)^3 ([G,O]d|GOGd)   
        
        #      = cos(x/2)^4 (Od|Od) + sin(x/2)^4 (GOGd|GOGd) + sin(x/2)^2cos(x/2)^2 ([G,O]d|[G,O]d)
        #      + 2 cos(x/2)^2 sin(x/2)^2 (Od|GOGd) 
        #      + 2i cos(x/2)^3 sin(x/2)^1 (Od|comd) 
        #      + 2i cos(x/2)^1 sin(x/2)^3 (GOGd|comd) 
        
        #      = cos(x/2)^4 A + sin(x/2)^4 B + sin(x/2)^2cos(x/2)^2 C
        #      + 2 cos(x/2)^2 sin(x/2)^2  D 
        #      + 2i cos(x/2)^3 sin(x/2)^1 E 
        #      + 2i cos(x/2)^1 sin(x/2)^3 F
        
        #       where:
        #           A = (Od|Od)
        #           B = (GOGd|GOGd)
        #           C = ([G,O]d|[G,O]d)
        #           D = (Od|GOGd)
        #           E = (Od|[G,O]d)
        #           F = (GOGd|[G,O]d)
        #      
        #       Eventually we can simplify:
        #       since (Od|Od) == (GOGd|GOGd)
        #      = (cos^4 + sin^4) (Od|Od)  
        #       + sin^2 * cos^2 ((comd|comd) + 2 (Od|GOGd) )
        #       + 2i cos^3sin (Od|comd)
        #       + 2i cos sin^3 (GOGd|comd)
        #      
        #       since (GOGd|comd) == -(Od|comd)
        #      = (cos^4 + sin^4) (Od|Od)  
        #       + sin^2 * cos^2 ((comd|comd) + 2 (Od|GOGd) )
        #       + 2i (cos^3sin - cos sin^3) (Od|comd)
        #      
        #      
        c = cos(θ/2)
        s = sin(θ/2)
        return c^4 * A + s^4 * B + s^2*c^2*(C+2D) + 2im *c^3*s*E + 2im *c*s^3*F
    end

    @show overlap(GOGd,Od) overlap(GOGd,GOGd) overlap(Od,Od)
    @show overlap(GOGd, comd) overlap(comd, GOGd) 
    @show overlap(Od, comd) overlap(comd, Od) 
    # Now do expectation value
    Oeval = expectation_value(O, ψ)
    OG = O*G
    OGeval = expectation_value(OG, ψ)
    GOGeval = expectation_value(G*O*G, ψ)
    function cost_eval(θ)
        # Cost function for <ψ| U(θ)' O U(θ)|ψ>
        return cos(θ/2)^2 * Oeval + sin(θ/2)^2 * GOGeval - 2im*cos(θ/2)*sin(θ/2)*OGeval
    end


    for i in 0:.1:1
        θ = i*2π
        U = TEO(θ)
        cost1_ref = norm(diag(evolve(O,G,θ)))^2
        # cost1_ref = norm(diag(U' * Omat * U))
        cost2_ref = expectation_value(evolve(O,G,θ), ψ)
        # cost2_ref = ψvec' * (U' * Omat * U) * ψvec

        cost1_test = cost_diag(θ)
        cost2_test = cost_eval(θ)

        @printf(" %5.2f %12.8f %12.8f %12.8f %12.8f\n", θ, real(cost1_ref), real(cost1_test), real(cost2_ref), real(cost2_test))

        # @show θ, cost2, cost_diag(θ/2) 
        # @show norm(diag(O))
        # @show norm(diag(evolve(O,G,θ)))
    end
end

test1()