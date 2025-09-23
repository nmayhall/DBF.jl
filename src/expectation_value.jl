function optimize_theta_expval(O::PauliSum{N,T}, G::PauliBasis{N}, ψ::Ket{N}; stepsize=.001, verbose=1) where {N,T}
    Oeval = expectation_value(O, ψ)
    OG = O*G
    OGeval = expectation_value(OG, ψ)
    GOGeval = expectation_value(G*O*G, ψ)
    function cost(θ)
        # Cost function for <ψ| U(θ)' O U(θ)|ψ>
        return real(cos(θ/2)^2 * Oeval + sin(θ/2)^2 * GOGeval - 2im*cos(θ/2)*sin(θ/2)*OGeval)
    end
    
    idx = argmin([cost(i*2π) for i in 0:stepsize:1-stepsize])
    # for i in 0:stepsize:1-stepsize
    #     @show i*2π, cost(i*2π)
    # end
    θ = (idx-1) * stepsize * 2π
    return θ, cost
end